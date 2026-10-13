"""Tests for Task 9 transformer code using a TINY randomly initialised BERT.

No pretrained model is downloaded: a 2-layer, 32-dim BERT and a small
WordPiece vocabulary are created in a temporary folder.
"""

import json

import joblib
import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

from src.data.eda import class_names
from src.inference import text as inference
from src.models import transformer_text as tt
from src.models.text_baselines import build_pipeline
from src.preprocessing import preprocess
from src.utils.config import REPO_ROOT, load_config

WORDS = ["ladki", "bhai", "nahi", "kar", "raha", "hai", "she", "exam", "mom", "dumb", "kitchen", "friend",
         "teacher", "beti", "class", "movie", "is", "so", "ko", "me", "user", "<", ">"]
TRAIN = ["ladki ko kitchen", "she is so dumb", "beti nahi", "she kitchen", "ladki dumb hai",
         "bhai exam hai", "mom movie", "teacher class", "friend exam", "bhai kar raha"] * 3
Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 3)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_bert")
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]), encoding="utf-8")
    tok = BertTokenizer(str(d / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64,
                     id2label={0: "non_misogynistic", 1: "misogynistic"},
                     label2id={"non_misogynistic": 0, "misogynistic": 1})
    torch.manual_seed(0)
    return tok, cfg


def classifier(cfg, seed=0):
    torch.manual_seed(seed)
    return BertForSequenceClassification(cfg)


def ft_cfg(**kw):
    base = dict(batch_size=8, max_epochs=3, learning_rate=5e-3, warmup_ratio=0.0, max_length=16, seed=1,
                early_stopping_patience=5)
    base.update(kw)
    return tt.FineTuneConfig(**base)


# ---------------- tokenization ----------------

def test_token_lengths_and_truncation(tiny):
    tok, _ = tiny
    lengths = tt.token_lengths(tok, ["ladki", "ladki ko kitchen hai"])
    assert lengths.tolist() == [3, 6]                       # includes [CLS] and [SEP]
    assert tt.truncation_rate(lengths, 4) == 0.5 and tt.truncation_rate(lengths, 6) == 0.0


# ---------------- frozen embeddings ----------------

def test_mean_pooled_embeddings_shape_and_mask_invariance(tiny):
    tok, cfg = tiny
    torch.manual_seed(0)
    enc = BertModel(cfg)
    alone = tt.mean_pooled_embeddings(["ladki"], tok, enc, max_length=16)
    padded = tt.mean_pooled_embeddings(["ladki", "ladki ko kitchen hai she is so dumb"], tok, enc, max_length=16)
    assert alone.shape == (1, 32) and padded.shape == (2, 32)
    np.testing.assert_allclose(alone[0], padded[0], atol=1e-5)      # padding does not change the embedding
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    tt.mean_pooled_embeddings(TRAIN, tok, enc, max_length=16)
    assert all(torch.equal(before[k], v) for k, v in enc.state_dict().items())   # encoder stays frozen


# ---------------- fine-tuning ----------------

def test_fine_tune_learns_selects_best_and_freezes_word_embeddings(tiny):
    tok, cfg = tiny
    model = classifier(cfg)
    emb_before = model.get_input_embeddings().weight.detach().clone()
    result = tt.fine_tune(model, tok, TRAIN, Y, TRAIN[:10], Y[:10], ft_cfg(), log=lambda *_: None)
    assert len(result.history) == 3
    assert result.best_val_macro_f1 == max(h.val_macro_f1 for h in result.history)
    assert result.history[result.best_epoch - 1].val_macro_f1 == result.best_val_macro_f1
    assert torch.equal(model.get_input_embeddings().weight, emb_before)      # frozen word embeddings
    assert result.trainable_parameters < result.total_parameters
    logits = tt.predict_logits(model, tok, TRAIN[:10], 16)
    assert logits.shape == (10, 2)                                           # binary head: 2 classes
    np.testing.assert_allclose(tt.softmax(logits).sum(1), 1.0, atol=1e-6)


def test_early_stopping(tiny):
    tok, cfg = tiny
    result = tt.fine_tune(classifier(cfg), tok, TRAIN, Y, TRAIN[:10], Y[:10],
                          ft_cfg(max_epochs=10, learning_rate=0.0, early_stopping_patience=1), log=lambda *_: None)
    assert len(result.history) == 2          # no improvement with lr=0 -> stops after patience


def test_fine_tune_deterministic(tiny):
    tok, cfg = tiny
    runs = []
    for _ in range(2):
        m = classifier(cfg)
        tt.fine_tune(m, tok, TRAIN, Y, TRAIN[:10], Y[:10], ft_cfg(max_epochs=2), log=lambda *_: None)
        runs.append(tt.predict_logits(m, tok, TRAIN[:10], 16))
    np.testing.assert_allclose(runs[0], runs[1], atol=1e-6)


def test_save_and_reload_reproduce_logits(tiny, tmp_path):
    tok, cfg = tiny
    model = classifier(cfg)
    tt.fine_tune(model, tok, TRAIN, Y, TRAIN[:10], Y[:10], ft_cfg(max_epochs=1), log=lambda *_: None)
    meta = {"label_map": {0: "non_misogynistic", 1: "misogynistic"}, "preprocessing_mode": "basic",
            "preprocessing_version": "1.0", "max_length": 16}
    tt.save_fine_tuned(model, tok, tmp_path / "ft", meta)
    model2, tok2, meta2 = tt.load_fine_tuned(tmp_path / "ft")
    np.testing.assert_allclose(tt.predict_logits(model, tok, TRAIN[:10], 16),
                               tt.predict_logits(model2, tok2, TRAIN[:10], 16), atol=1e-5)
    assert meta2["preprocessing_version"] == "1.0" and model2.config.id2label[1] == "misogynistic"


# ---------------- inference interface ----------------

@pytest.fixture
def registry(tiny, tmp_path, monkeypatch):
    tok, cfg = tiny
    monkeypatch.setattr(inference, "model_dir", lambda config=None: tmp_path)
    inference._load.cache_clear()
    model = classifier(cfg)
    tt.fine_tune(model, tok, TRAIN, Y, TRAIN[:10], Y[:10], ft_cfg(max_epochs=1), log=lambda *_: None)
    tt.save_fine_tuned(model, tok, tmp_path / "ft", {"preprocessing_mode": "hinglish_normalized"})
    config = load_config()
    pipe = build_pipeline("logistic_regression", "basic", config, 42).fit(TRAIN, Y)
    joblib.dump(pipe, tmp_path / "lr.joblib")
    reg = {"models": {
        "FT": {"kind": "fine_tuned", "artifact": "{MODEL_DIR}/ft", "preprocessing_mode": "hinglish_normalized",
               "preprocessing_version": "1.0", "max_length": 16},
        "LR": {"kind": "tfidf_logreg", "artifact": "{MODEL_DIR}/lr.joblib", "preprocessing_mode": "basic",
               "preprocessing_version": "1.0"}}}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(reg), encoding="utf-8")
    yield path
    inference._load.cache_clear()


@pytest.mark.parametrize("version", ["FT", "LR"])
def test_predict_text_structure_and_honest_scores(registry, version):
    out = inference.predict_text("Ye LADKI kitchen me @someone", version, registry_path=registry)
    assert out["predicted_label"] in (0, 1) and out["predicted_name"] in ("non_misogynistic", "misogynistic")
    assert out["calibrated"] is False and "uncalibrated" in out["score_type"]
    assert 0.0 <= out["score"] <= 1.0
    assert out["model_version"] == version and out["preprocessing_version"] == "1.0"


def test_predict_text_uses_task6_preprocessing(registry, monkeypatch):
    seen = []
    real = tt.preprocess_all
    monkeypatch.setattr(tt, "preprocess_all", lambda texts, mode, opts=None: seen.append(mode) or real(texts, mode, opts))
    inference.predict_text("kr rha hu", "FT", registry_path=registry)
    assert seen == ["hinglish_normalized"]
    assert preprocess("kr rha hu", "hinglish_normalized") == "kar raha hoon"


def test_predict_text_empty_and_none(registry):
    assert inference.predict_text("", "FT", registry_path=registry)["processed_text_empty"]
    assert inference.predict_text(None, "LR", registry_path=registry)["processed_text_empty"]


def test_predict_text_deterministic(registry):
    a = inference.predict_text("she is so dumb", "FT", registry_path=registry)
    b = inference.predict_text("she is so dumb", "FT", registry_path=registry)
    assert a == b


def test_svm_scores_are_not_called_probabilities():
    assert "NOT a probability" in inference.SCORE_TYPES["tfidf_svm"]


# ---------------- label mapping, config, test-set protection ----------------

def test_label_mapping_matches_config():
    assert tt.LABEL_NAMES == inference.LABEL_NAMES == class_names(load_config())


def test_transformer_config_is_untuned_baseline():
    t = load_config()["text_model"]["transformer"]
    assert t["checkpoint"] == "google/muril-base-cased" and t["fine_tune"]["selection_metric"] == "macro_f1"
    assert set(t["experiments"]) == {"T01", "T02", "T03", "T04", "T05"}


def test_task9_scripts_never_touch_test_split():
    for rel in ("scripts/train_text_transformers.py", "scripts/report_text_models.py",
                "src/models/transformer_text.py", "src/inference/text.py"):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'load_split("test"' not in source and "allow_test=True" not in source, rel


# ---------------- report helper: descriptive sub-label recall ----------------

def test_subcategory_recall(tmp_path, monkeypatch):
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location("report_text_models", REPO_ROOT / "scripts/report_text_models.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    (tmp_path / "metrics").mkdir()
    (tmp_path / "results/tables/text_baselines").mkdir(parents=True)
    monkeypatch.setattr(report, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)
    val = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "target_label": [1, 1, 1, 0],
                        "Objectification": [1, 1, 0, 0], "Prejudice": [0, 1, 1, 0], "Humiliation": [0, 0, 0, 0]})
    pd.DataFrame({"sample_id": ["d", "c", "b", "a"], "pred": [0, 1, 0, 1]}).to_csv(tmp_path / "metrics/T05_validation_predictions.csv")
    pd.DataFrame({"sample_id": ["b"]}).to_csv(tmp_path / "results/tables/text_baselines/validation_errors.csv")
    table = report.subcategory_recall(val, {"T05": {}}).set_index("group")
    assert table.loc["Objectification", "recall_T05"] == 0.5          # a found, b missed
    assert table.loc["Prejudice", "recall_T05"] == 0.5                # c found, b missed
    assert table.loc["any misogynistic", "validation_samples"] == 3
    assert table.loc["multiple sub-labels", "validation_samples"] == 1
    assert table.loc["Objectification", "recall_B03"] == 0.5          # B03 wrong only on b
    assert table.loc["any misogynistic", "recall_B03"] == round(2 / 3, 4)
