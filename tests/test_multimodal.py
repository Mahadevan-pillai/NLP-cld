"""Tests for Task 11 multimodal code: synthetic features, a tiny random BERT and a tiny random SigLIP.

No pretrained model is downloaded and no dataset file is needed.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (BertConfig, BertForSequenceClassification, BertModel, BertTokenizer,
                          SiglipImageProcessorPil, SiglipVisionConfig, SiglipVisionModel)

from src.data.eda import class_names
from src.evaluation.metrics import classification_metrics
from src.features import multimodal as mm
from src.inference import multimodal as inference
from src.models import fusion
from src.models.image_encoder import LABEL_NAMES, image_embeddings
from src.preprocessing.image import IMAGE_PREPROCESSING_VERSION, ImageLoadError
from src.utils.config import REPO_ROOT, load_config


def synthetic(n=200, d_text=12, d_image=10, seed=0):
    """Labels depend on BOTH modalities: text carries half the signal, image the other half."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    text = rng.normal(size=(n, d_text)).astype(np.float32)
    image = rng.normal(size=(n, d_image)).astype(np.float32) * 50 + 100        # different scale on purpose
    text[:, 0] += 2.5 * (y - 0.5)
    image[:, 0] += 120 * (y - 0.5)
    return text, image, y


def small_cfg(**kw):
    base = dict(projection_dim=8, hidden_dim=8, dropout=0.1, learning_rate=1e-2, batch_size=32, max_epochs=15,
                early_stopping_patience=20, seed=3)
    base.update(kw)
    return fusion.FusionConfig(**base)


# ---------------- alignment ----------------

def test_align_joins_on_sample_id_not_row_order():
    labels = pd.DataFrame({"sample_id": ["a", "b", "c"], "target_label": ["1", "0", "1"]})
    text = (np.array(["c", "a", "b"]), np.array([[3.0], [1.0], [2.0]], dtype=np.float32))
    image = (np.array(["b", "c", "a", "zz"]), np.array([[20.0], [30.0], [10.0], [99.0]], dtype=np.float32))
    out, aligned, audit = mm.align(labels, {"text": text, "image": image})
    assert aligned["text"][:, 0].tolist() == [1.0, 2.0, 3.0] and aligned["image"][:, 0].tolist() == [10.0, 20.0, 30.0]
    assert out["sample_id"].tolist() == ["a", "b", "c"]
    image_audit = next(r for r in audit if r["modality"] == "image")
    assert image_audit["extra_ids_not_in_labels"] == 1 and not image_audit["row_order_already_identical"]


@pytest.mark.parametrize("ids, values", [
    (["a", "a", "b"], [[1.0], [1.0], [2.0]]),          # duplicate id
    (["a", "b"], [[1.0], [2.0]]),                      # labelled sample "c" missing
    (["a", "b", "c"], [[1.0], [np.nan], [2.0]]),       # non-finite value
])
def test_align_rejects_duplicates_missing_and_nan(ids, values):
    labels = pd.DataFrame({"sample_id": ["a", "b", "c"], "target_label": ["1", "0", "1"]})
    with pytest.raises(ValueError):
        mm.align(labels, {"text": (np.array(ids), np.array(values, dtype=np.float32))})


def test_feature_cache_roundtrip_and_stale_detection(tmp_path):
    ids, feats = ["s1", "s2"], np.arange(6, dtype=np.float32).reshape(2, 3)
    meta = {"representation": "x", "inputs_fingerprint": mm.inputs_fingerprint(ids, ["t1", "t2"])}
    mm.save_features(tmp_path / "f", ids, feats, meta)
    got_ids, got = mm.load_features(tmp_path / "f", meta)
    assert got_ids.tolist() == ids and np.array_equal(got, feats)
    assert mm.load_features(tmp_path / "f", {**meta, "inputs_fingerprint": mm.inputs_fingerprint(ids, ["t1", "CHANGED"])}) is None
    assert mm.load_features(tmp_path / "missing", meta) is None
    assert mm.inputs_fingerprint(["a", "b"], ["x", "y"]) != mm.inputs_fingerprint(["b", "a"], ["y", "x"])
    with pytest.raises(ValueError):
        mm.save_features(tmp_path / "bad", ["only-one"], feats, meta)


# ---------------- fusion model ----------------

def test_standardizer_uses_train_statistics_only():
    tr = np.array([[0.0, 5.0], [2.0, 5.0]], dtype=np.float32)
    s = fusion.Standardizer().fit(tr)
    assert s.mean.tolist() == [1.0, 5.0] and s.std.tolist() == [1.0, 1.0]          # constant column -> std 1
    assert s.transform(np.array([[3.0, 5.0]], dtype=np.float32)).tolist() == [[2.0, 0.0]]


def test_forward_shape():
    model = fusion.FusionMLP(12, 10, small_cfg())
    out = model(torch.zeros(5, 12), torch.zeros(5, 10))
    assert tuple(out.shape) == (5, 2)                                           # binary head: 2 logits


def test_training_learns_selects_best_epoch_and_uses_both_modalities():
    text, image, y = synthetic()
    model, stds, result = fusion.train_fusion(text[:150], image[:150], y[:150], text[150:], image[150:], y[150:],
                                              small_cfg(), log=lambda *_: None)
    proba = fusion.predict_proba(model, stds, text[150:], image[150:])
    assert proba.shape == (50, 2) and np.allclose(proba.sum(1), 1.0, atol=1e-6)
    f1 = classification_metrics(y[150:], proba.argmax(1), {0: "a", 1: "b"})["macro_f1"]
    assert f1 > 0.8
    assert result.best_val_macro_f1 == max(h.val_macro_f1 for h in result.history)
    assert np.allclose(stds[0].mean, text[:150].mean(0), atol=1e-5)           # TRAIN statistics, not validation
    no_image = fusion.predict_proba(model, stds, text[150:], np.repeat(stds[1].mean[None], 50, 0)).argmax(1)
    assert classification_metrics(y[150:], no_image, {0: "a", 1: "b"})["macro_f1"] < f1   # image signal is used


def test_early_stopping_and_determinism():
    text, image, y = synthetic(n=120)
    _, _, r = fusion.train_fusion(text[:90], image[:90], y[:90], text[90:], image[90:], y[90:],
                                  small_cfg(learning_rate=0.0, early_stopping_patience=2, max_epochs=30), log=lambda *_: None)
    assert len(r.history) == 3                                                  # no improvement with lr=0
    runs = []
    for _ in range(2):
        m, s, _ = fusion.train_fusion(text[:90], image[:90], y[:90], text[90:], image[90:], y[90:], small_cfg(max_epochs=4),
                                      log=lambda *_: None)
        runs.append(fusion.predict_proba(m, s, text[90:], image[90:]))
    np.testing.assert_allclose(runs[0], runs[1], atol=1e-6)


def test_save_load_reproduces_predictions_without_pickle(tmp_path):
    text, image, y = synthetic(n=100)
    model, stds, _ = fusion.train_fusion(text[:80], image[:80], y[:80], text[80:], image[80:], y[80:], small_cfg(max_epochs=3),
                                         log=lambda *_: None)
    fusion.save_fusion(model, stds, small_cfg(max_epochs=3), tmp_path / "m", {"label_map": {"0": "non_misogynistic"}})
    assert sorted(p.name for p in (tmp_path / "m").iterdir()) == ["fusion.safetensors", "fusion_config.json"]
    m2, s2, meta = fusion.load_fusion(tmp_path / "m")
    np.testing.assert_allclose(fusion.predict_proba(model, stds, text[80:], image[80:]),
                               fusion.predict_proba(m2, s2, text[80:], image[80:]), atol=1e-6)
    assert meta["text_dim"] == 12 and meta["image_dim"] == 10


# ---------------- text features from a tiny fine-tuned BERT ----------------

WORDS = ["ladki", "kitchen", "dumb", "bhai", "exam", "movie", "she", "hai", "user", "<", ">"]


@pytest.fixture(scope="module")
def tiny_text(tmp_path_factory):
    d = tmp_path_factory.mktemp("bert")
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]), encoding="utf-8")
    tok = BertTokenizer(str(d / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=32)
    torch.manual_seed(0)
    return tok, BertForSequenceClassification(cfg).eval(), BertModel(cfg).eval()


def test_finetuned_pooled_features_feed_the_classifier_head(tiny_text):
    tok, clf_model, _ = tiny_text
    texts = ["ladki kitchen", "bhai exam hai", "she dumb", ""]
    feats = mm.finetuned_pooled_features(texts, clf_model, tok, max_length=16, batch_size=3)
    assert feats.shape == (4, 16) and feats.dtype == np.float32
    with torch.no_grad():
        direct = clf_model(**tok(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")).logits.numpy()
        via_head = clf_model.classifier(torch.from_numpy(feats)).numpy()
    np.testing.assert_allclose(direct, via_head, atol=1e-5)                    # exactly the head's input


# ---------------- inference interface ----------------

@pytest.fixture
def registry(tiny_text, tmp_path, monkeypatch):
    tok, clf_model, base = tiny_text
    processor = SiglipImageProcessorPil(size={"height": 32, "width": 32}, resample=3, do_rescale=True,
                                        rescale_factor=1 / 255, do_normalize=True,
                                        image_mean=[0.5] * 3, image_std=[0.5] * 3)
    torch.manual_seed(0)
    vision = SiglipVisionModel(SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                                  num_attention_heads=2, image_size=32, patch_size=8)).eval()
    monkeypatch.setattr(inference, "_text_encoder",
                        lambda rep, source, rev: (clf_model, tok) if rep == "finetuned_pooled" else (base, tok))
    monkeypatch.setattr(inference, "_image_encoder", lambda ckpt, rev: (processor, vision))
    monkeypatch.setattr(inference, "REPO_ROOT", tmp_path)
    inference._classifier.cache_clear()
    rng = np.random.default_rng(0)
    text, image, y = rng.normal(size=(40, 16)).astype(np.float32), rng.normal(size=(40, 16)).astype(np.float32), np.arange(40) % 2
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(np.hstack([text, image]), y)
    joblib.dump(lr, tmp_path / "m01.joblib")
    model, stds, _ = fusion.train_fusion(text, image, y, text, image, y, small_cfg(max_epochs=2), log=lambda *_: None)
    fusion.save_fusion(model, stds, small_cfg(max_epochs=2), tmp_path / "m02", {})
    common = {"text_source": "tiny", "text_encoder_revision": "r", "text_preprocessing_mode": "hinglish_normalized",
              "text_preprocessing_version": "1.0", "max_length": 16, "image_encoder": "tiny", "image_encoder_revision": "r",
              "image_preprocessing_version": IMAGE_PREPROCESSING_VERSION}
    models = {"M01": {**common, "kind": "concat_logreg", "artifact": "m01.joblib", "text_representation": "finetuned_pooled"},
              "M02": {**common, "kind": "fusion_mlp", "artifact": "m02", "text_representation": "finetuned_pooled"},
              "M03": {**common, "kind": "fusion_mlp", "artifact": "m02", "text_representation": "frozen_mean"}}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    yield path
    inference._classifier.cache_clear()


IMG = Image.new("RGB", (40, 30), (200, 20, 20))


@pytest.mark.parametrize("version", ["M01", "M02", "M03"])
def test_predict_multimodal_structure_and_honest_scores(registry, version):
    out = inference.predict_multimodal(IMG, "ye ladki kitchen me", version, registry_path=registry)
    assert out["predicted_label"] in (0, 1) and out["predicted_name"] == LABEL_NAMES[out["predicted_label"]]
    assert out["modality"] == "image+text" and out["calibrated"] is False and out["ocr_used"] is False
    assert "uncalibrated" in out["score_type"] and 0.0 <= out["score"] <= 1.0
    assert out == inference.predict_multimodal(IMG, "ye ladki kitchen me", version, registry_path=registry)   # deterministic


def test_predict_multimodal_missing_and_bad_inputs(registry):
    with pytest.raises(inference.MissingModalityError):
        inference.predict_multimodal(None, "text", "M02", registry_path=registry)
    with pytest.raises(inference.MissingModalityError):
        inference.predict_multimodal(IMG, None, "M02", registry_path=registry)
    with pytest.raises(inference.MissingModalityError):
        inference.predict_multimodal(IMG, 123, "M02", registry_path=registry)
    with pytest.raises(ImageLoadError):
        inference.predict_multimodal(b"not an image", "text", "M02", registry_path=registry)
    empty = inference.predict_multimodal(IMG, "   ", "M02", registry_path=registry)         # allowed, flagged
    assert empty["processed_text_empty"] is True


def test_predict_multimodal_matches_offline_pipeline(registry, tiny_text):
    tok, clf_model, _ = tiny_text
    t = mm.finetuned_pooled_features(["she dumb"], clf_model, tok, 16)
    processor, vision = inference._image_encoder("tiny", "r")
    i = image_embeddings([IMG], processor, vision)
    model, stds, _ = fusion.load_fusion(registry.parent / "m02")
    expected = fusion.predict_proba(model, stds, t, i)[0, 1]
    assert inference.predict_multimodal(IMG, "she dumb", "M02", registry_path=registry)["score"] == round(float(expected), 6)


# ---------------- config, labels, test-set protection ----------------

def test_multimodal_config():
    m = load_config()["multimodal_model"]
    assert m["experiments"] == {"M01": ["logistic_regression", "finetuned_pooled"],
                                "M02": ["fusion_mlp", "finetuned_pooled"], "M03": ["fusion_mlp", "frozen_mean"]}
    assert m["selection_metric"] == "macro_f1" and m["fusion_mlp"]["projection_dim"] == 256


def test_label_mapping_consistent():
    assert LABEL_NAMES == class_names(load_config())


def test_task11_sources_never_touch_test_split():
    for rel in ("scripts/train_multimodal.py", "scripts/report_multimodal.py", "src/features/multimodal.py",
                "src/models/fusion.py", "src/inference/multimodal.py"):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'load_split("test"' not in source and "allow_test=True" not in source, rel
