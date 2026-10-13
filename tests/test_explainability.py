"""Tests for Task 15 explainability code. Synthetic fixtures only: a tiny random BERT, a tiny random SigLIP,
a fake encoder and a fake OCR reader. No pretrained model is downloaded and no dataset file is needed;
no expensive research explanation is regenerated here.
"""

import json

import joblib
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer, SiglipImageProcessorPil,
                          SiglipVisionConfig, SiglipVisionModel)

from src.explainability import errors as er
from src.explainability import explanation as ex
from src.explainability import image as xi
from src.explainability import moderation as mod
from src.explainability import multimodal as xm
from src.explainability import text as xt
from src.inference import image as image_inference
from src.inference import multimodal as mm_inference
from src.inference import text as text_inference
from src.inference.multimodal import MissingModalityError
from src.models import fusion
from src.models import image_encoder as ie
from src.models import transformer_text as tt
from src.models.text_baselines import build_pipeline
from src.preprocessing import PreprocessingConfig, preprocess
from src.preprocessing.image import IMAGE_PREPROCESSING_VERSION, ImageLoadError
from src.utils.config import REPO_ROOT, load_config

WORDS = ["ladki", "bhai", "nahi", "kar", "raha", "hai", "she", "exam", "mom", "dumb", "kitchen", "friend",
         "teacher", "beti", "class", "movie", "is", "so", "ko", "me", "user", "rha", "hu", "<", ">", "##s"]
TRAIN = ["ladki ko kitchen", "she is so dumb", "beti nahi", "she kitchen", "ladki dumb hai",
         "bhai exam hai", "mom movie", "teacher class", "friend exam", "bhai kar raha"] * 3
Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 3)


def synthetic_image(color, size=(48, 40)):
    return Image.new("RGB", size, color)


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_bert")
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]), encoding="utf-8")
    tok = BertTokenizer(str(d / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64,
                     id2label={0: "non_misogynistic", 1: "misogynistic"}, label2id={"non_misogynistic": 0, "misogynistic": 1})
    torch.manual_seed(0)
    model = BertForSequenceClassification(cfg).eval()
    return tok, model


@pytest.fixture(scope="module")
def tiny_siglip():
    processor = SiglipImageProcessorPil(size={"height": 32, "width": 32}, resample=3, do_rescale=True,
                                        rescale_factor=1 / 255, do_normalize=True,
                                        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    torch.manual_seed(0)
    model = SiglipVisionModel(SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                                 num_attention_heads=2, image_size=32, patch_size=8)).eval()
    return processor, model


@pytest.fixture
def registries(tiny_bert, tiny_siglip, tmp_path, monkeypatch):
    """Text (FT + LR), image (I01) and multimodal (M02) registries over tiny models, wired like the real ones."""
    tok, bert = tiny_bert
    processor, vision = tiny_siglip
    for fn in (text_inference._load, image_inference._load, mm_inference._classifier):
        fn.cache_clear()
    # text
    monkeypatch.setattr(text_inference, "model_dir", lambda config=None: tmp_path)
    tt.save_fine_tuned(bert, tok, tmp_path / "ft", {"preprocessing_mode": "hinglish_normalized"})
    pipe = build_pipeline("logistic_regression", "basic", load_config(), 42).fit(TRAIN, Y)
    joblib.dump(pipe, tmp_path / "lr.joblib")
    text_reg = tmp_path / "text_registry.json"
    text_reg.write_text(json.dumps({"models": {
        "FT": {"kind": "fine_tuned", "artifact": "{MODEL_DIR}/ft", "preprocessing_mode": "hinglish_normalized",
               "preprocessing_version": "1.0", "max_length": 16},
        "LR": {"kind": "tfidf_logreg", "artifact": "{MODEL_DIR}/lr.joblib", "preprocessing_mode": "basic",
               "preprocessing_version": "1.0"}}}), encoding="utf-8")
    # image
    monkeypatch.setattr(image_inference, "_encoder", lambda checkpoint, revision: (processor, vision))
    monkeypatch.setattr(image_inference, "REPO_ROOT", tmp_path)
    imgs = [synthetic_image((200 + i, 10, 10)) for i in range(6)] + [synthetic_image((10, 10, 200 + i)) for i in range(6)]
    emb = ie.image_embeddings(imgs, processor, vision, batch_size=4)
    clf = ie.build_classifier("logistic_regression", load_config()["image_model"]["logistic_regression"], 42)
    joblib.dump(clf.fit(emb, [1] * 6 + [0] * 6), tmp_path / "I01.joblib")
    image_reg = tmp_path / "image_registry.json"
    image_reg.write_text(json.dumps({"models": {"I01": {
        "kind": "frozen_logreg", "artifact": "I01.joblib", "encoder": "tiny/siglip", "encoder_revision": "r" * 40,
        "image_preprocessing_version": IMAGE_PREPROCESSING_VERSION}}}), encoding="utf-8")
    # multimodal (fusion over the tiny BERT pooled output + tiny SigLIP embedding)
    monkeypatch.setattr(mm_inference, "_text_encoder", lambda rep, source, rev: (bert, tok))
    monkeypatch.setattr(mm_inference, "_image_encoder", lambda ckpt, rev: (processor, vision))
    monkeypatch.setattr(mm_inference, "REPO_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    t_tr, i_tr = rng.normal(size=(40, 32)).astype(np.float32), rng.normal(size=(40, 32)).astype(np.float32)
    y = np.arange(40) % 2
    cfg = fusion.FusionConfig(projection_dim=8, hidden_dim=8, dropout=0.1, learning_rate=1e-2, batch_size=16,
                              max_epochs=2, early_stopping_patience=5, seed=3)
    model, stds, _ = fusion.train_fusion(t_tr, i_tr, y, t_tr, i_tr, y, cfg, log=lambda *_: None)
    fusion.save_fusion(model, stds, cfg, tmp_path / "m02", {})
    mm_reg = tmp_path / "mm_registry.json"
    mm_reg.write_text(json.dumps({"models": {"M02": {
        "kind": "fusion_mlp", "artifact": "m02", "text_representation": "finetuned_pooled", "text_source": "tiny",
        "text_encoder_revision": "r", "text_preprocessing_mode": "hinglish_normalized", "text_preprocessing_version": "1.0",
        "max_length": 16, "image_encoder": "tiny", "image_encoder_revision": "r",
        "image_preprocessing_version": IMAGE_PREPROCESSING_VERSION}}}), encoding="utf-8")
    monkeypatch.setattr(text_inference, "REGISTRY_PATH", text_reg)
    monkeypatch.setattr(image_inference, "REGISTRY_PATH", image_reg)
    monkeypatch.setattr(mm_inference, "REGISTRY_PATH", mm_reg)
    yield {"text": text_reg, "image": image_reg, "multimodal": mm_reg, "tmp": tmp_path}
    for fn in (text_inference._load, image_inference._load, mm_inference._classifier):
        fn.cache_clear()


# ---------------- text ----------------

def test_merge_wordpieces_sums_pieces_skips_specials_and_flags_unknown():
    out = xt.merge_wordpieces(["[CLS]", "double", "##me", "x", "[UNK]", "[SEP]"], np.array([9, 1.0, 0.5, -2, 0.3, 9]), "[UNK]")
    assert [w["token"] for w in out] == ["doubleme", "x", "[UNK]"]
    assert out[0]["contribution"] == 1.5 and out[0]["wordpieces"] == 2 and out[0]["direction"] == "towards misogynistic"
    assert out[1]["direction"] == "towards non-misogynistic" and out[2]["unknown_token"] is True


def test_integrated_gradients_completeness_and_model_consistency(tiny_bert):
    tok, model = tiny_bert
    r = xt.transformer_attribution(model, tok, "ladki ko kitchen me", 16, steps=128)
    assert [w["token"] for w in r["tokens"]] == ["ladki", "ko", "kitchen", "me"]
    assert abs(r["completeness_error"]) < 1e-2 * max(1.0, abs(r["logit_difference"] - r["baseline_logit_difference"]))
    with torch.no_grad():
        logits = model(**tok(["ladki ko kitchen me"], return_tensors="pt")).logits[0]
    assert r["logit_difference"] == pytest.approx(float(logits[1] - logits[0]), abs=1e-5)   # same forward as inference


def test_explain_text_transformer_matches_inference_and_serialises(registries):
    e = xt.explain_text("Ye LADKI kitchen me", "FT", steps=64)
    p = text_inference.predict_text("Ye LADKI kitchen me", "FT", registries["text"])
    assert (e["predicted_label"], e["score"]) == (p["predicted_label"], p["score"])
    assert e["predicted_label"] == int(e["logit_difference"] > 0) and e["calibrated"] is False
    assert e["method"]["name"] == "integrated_gradients" and e["tokens"] and len(e["top_positive"]) <= 5
    assert all(t["contribution"] > 0 for t in e["top_positive"]) and all(t["contribution"] < 0 for t in e["top_negative"])
    assert any("not proof" in x for x in e["limitations"])
    json.dumps(e)


def test_explain_text_linear_is_exact_and_marks_unknown_words(registries):
    e = xt.explain_text("ladki kitchen zzzunknown", "LR")
    pipe = joblib.load(registries["tmp"] / "lr.joblib")
    assert e["decomposition_error"] == pytest.approx(0, abs=1e-9)
    assert e["decision_function"] == pytest.approx(float(pipe.decision_function(["ladki kitchen zzzunknown"])[0]), abs=1e-6)
    assert sum(f["contribution"] for f in e["features"]) + e["intercept"] == pytest.approx(e["decision_function"], abs=1e-5)
    unknown = next(t for t in e["tokens"] if t["token"] == "zzzunknown")
    assert unknown["not_in_vocabulary"] and unknown["contribution"] == 0.0
    assert e["predicted_label"] == text_inference.predict_text("ladki kitchen zzzunknown", "LR", registries["text"])["predicted_label"]


def test_explain_text_missing_and_empty_input(registries):
    with pytest.raises(ValueError):
        xt.explain_text(None, "FT")
    with pytest.raises(TypeError):
        xt.explain_text(5, "FT")
    e = xt.explain_text("   ", "FT")
    assert e["processed_text_empty"] and e["tokens"] == [] and any("empty" in x for x in e["limitations"])


def test_explanation_of_ocr_text_uses_the_models_task6_preprocessing(registries, tiny_bert):
    tok, _ = tiny_bert
    ocr_like = "kr rha hu @someone"
    processed = preprocess(ocr_like, "hinglish_normalized", PreprocessingConfig())
    expected = [w["token"] for w in xt.merge_wordpieces(tok.convert_ids_to_tokens(tok([processed])["input_ids"][0]),
                                                         np.zeros(64), tok.unk_token)]
    assert [w["token"] for w in xt.explain_text(ocr_like, "FT", steps=8)["tokens"]] == expected
    assert processed != ocr_like                                             # normalisation really applied


# ---------------- image ----------------

def test_occlusion_geometry():
    with pytest.raises(ValueError):
        xi.window_positions(224, 50, 28)
    pos = xi.window_positions(224, 56, 28)
    assert len(pos) == 49 and len(xi.window_positions(224, 112, 56)) == 9
    cells = xi.windows_to_cells(np.ones(len(pos)), pos, 224, 56, 28)
    assert cells.shape == (8, 8) and np.allclose(cells, 1.0)


class _QuadrantEncoder:
    """Fake vision encoder: the 'embedding' is the mean of the top-left 16 x 16 pixels."""
    def __call__(self, pixel_values):
        return type("Out", (), {"pooler_output": pixel_values[:, :1, :16, :16].mean(dim=(2, 3))})()


class _Identity:
    def decision_function(self, x):
        return x[:, 0]


def test_occlusion_attributes_only_the_informative_region():
    pv = torch.zeros(1, 3, 32, 32)
    pv[:, :, :16, :16] = 1.0
    full, occ, pos = xi.occlusion_embeddings(pv, _QuadrantEncoder(), window=8, stride=4)
    cell_map = xi.occlusion_map(float(_Identity().decision_function(full)[0]), _Identity().decision_function(occ), pos, 32, 8, 4)
    assert cell_map.shape == (8, 8) and (cell_map[:4, :4] > 0).all() and np.allclose(cell_map[5:, 5:], 0)
    assert xi.top_regions(cell_map, 1)[0]["box_fraction"][2] <= 0.5


def test_explain_image_structure_consistency_and_original_untouched(registries, tmp_path):
    path = tmp_path / "meme.png"
    synthetic_image((220, 30, 30)).save(path)
    before = path.read_bytes()
    e = xi.explain_image(path, "I01", window=8, stride=4)
    p = image_inference.predict_image(path, "I01", registries["image"])
    assert (e["predicted_label"], e["score"]) == (p["predicted_label"], p["score"])
    assert e["heatmap_shape"] == [8, 8] and len(e["heatmap"]) == 8 and e["method"]["forward_passes"] == 50
    assert e["predicted_label"] == int(e["logit_difference"] > 0) and e["text_used"] is False
    over = xi.overlay(path, np.array(e["heatmap"]))
    assert over.size == (48, 40) and path.read_bytes() == before
    json.dumps(e)
    with pytest.raises(ValueError):
        xi.explain_image(None, "I01")
    with pytest.raises(ImageLoadError):
        xi.explain_image(b"not an image", "I01", window=8, stride=4)


def test_region_share_and_concentration():
    m = np.zeros((8, 8))
    m[:4, :4] = 1.0
    r = xi.region_share(m, [(0, 0, 0.5, 0.5)])
    assert r == {"area_share": 0.25, "positive_mass_share": 1.0, "enrichment": 4.0}
    assert xi.region_share(np.zeros((8, 8)), [(0, 0, 1, 1)])["enrichment"] is None
    assert xi.concentration(m, 0.25) == 1.0 and xi.concentration(np.zeros((8, 8))) is None


# ---------------- multimodal ----------------

def test_modality_agreement_and_dependence_rules():
    assert xm.modality_agreement(1, 1, 1) == "AGREE" and xm.modality_agreement(0, 1, 1) == "PARTIAL DISAGREEMENT"
    assert xm.modality_agreement(0, 0, 1) == "DISAGREE"
    assert xm.dependence(1, 1, 1).startswith("label unchanged") and xm.dependence(1, 0, 1) == "label depends on the text"
    assert xm.dependence(1, 1, 0) == "label depends on the image" and xm.dependence(1, 0, 0) == "label needs both modalities"


def test_ablation_replaces_a_modality_by_its_train_mean():
    rng = np.random.default_rng(1)
    t, i = rng.normal(size=(30, 6)).astype(np.float32), rng.normal(size=(30, 5)).astype(np.float32) * 10 + 3
    cfg = fusion.FusionConfig(projection_dim=4, hidden_dim=4, dropout=0.0, learning_rate=1e-2, batch_size=8,
                              max_epochs=2, early_stopping_patience=5, seed=0)
    model, stds, _ = fusion.train_fusion(t, i, np.arange(30) % 2, t, i, np.arange(30) % 2, cfg, log=lambda *_: None)
    a = xm.ablation(model, stds, t[:1], i[:1])
    with torch.no_grad():                                            # train mean == zero after standardisation
        z = model(torch.zeros(1, 6), torch.from_numpy(stds[1].transform(i[:1]))).numpy()[0]
    assert a["text_neutralised"]["logit_difference"] == pytest.approx(float(z[1] - z[0]), abs=1e-5)
    full = fusion.predict_proba(model, stds, t[:1], i[:1])[0]
    assert a["full"]["predicted_label"] == int(np.argmax(full)) and a["full"]["score"] == pytest.approx(float(full[1]), abs=1e-5)
    assert a["label_changes_without_text"] == (a["text_neutralised"]["predicted_label"] != a["full"]["predicted_label"])


def test_explain_multimodal_consistency_attributions_and_missing_inputs(registries):
    img = synthetic_image((210, 20, 20))
    e = xm.explain_multimodal(img, "ladki kitchen me", "M02", "FT", "I01", attributions=True, ig_steps=64, window=8, stride=4)
    p = mm_inference.predict_multimodal(img, "ladki kitchen me", "M02", registries["multimodal"])
    assert e["consistent_with_inference"] and e["predictions"]["multimodal"]["score"] == p["score"]
    labels = [e["predictions"][k]["predicted_label"] for k in ("text", "image", "multimodal")]
    assert e["modality_agreement"] == xm.modality_agreement(*labels)
    att = e["attributions_through_fusion"]
    assert len(att["image"]["heatmap"]) == 8 and att["text"]["logit_difference"] == pytest.approx(
        e["ablation"]["full"]["logit_difference"], abs=1e-4)                 # IG forward == fusion forward
    assert abs(att["text"]["completeness_error"]) < 0.05
    json.dumps(e)
    for image, text in ((None, "x"), (img, None), (img, 3)):
        with pytest.raises(MissingModalityError):
            xm.explain_multimodal(image, text, "M02", "FT", "I01")


# ---------------- moderation ----------------

def test_moderation_rules_are_deterministic_and_never_block():
    policy = mod.load_policy()
    mod.validate_policy(policy)
    r = lambda **x: mod.recommend(x, "meme", policy)["recommendation"]          # noqa: E731
    assert r(text_label=1, image_label=1, multimodal_label=1) == "WARNING"
    assert r(text_label=0, image_label=0, multimodal_label=0) == "ALLOW"
    assert r(text_label=1, image_label=0, multimodal_label=0) == "HUMAN_REVIEW"
    assert r(ocr_route="reject") is None
    assert r(ocr_route="image_only_and_human_review", image_label=0) == "HUMAN_REVIEW"
    assert r(ocr_route="image_only", image_label=1) == "HUMAN_REVIEW" and r(ocr_route="image_only", image_label=0) == "ALLOW"
    t = lambda **x: mod.recommend(x, "text", policy)                              # noqa: E731
    assert t(text_label=1, second_text_label=1, processed_text_empty=True)["rule_id"] == "T0"
    assert t(text_label=1, second_text_label=1)["recommendation"] == "WARNING"
    assert t(text_label=1, second_text_label=0)["recommendation"] == "HUMAN_REVIEW"
    assert mod.recommend({"text_label": 0, "image_label": 0, "multimodal_label": 0})["automatic_block"] is False
    blocked = {**policy, "meme_mode": [{**policy["meme_mode"][0], "then": "BLOCK"}, *policy["meme_mode"][1:]]}
    with pytest.raises(ValueError):
        mod.validate_policy(blocked)
    with pytest.raises(ValueError):
        mod.validate_policy({**policy, "text_mode": [{"id": "T9", "then": "ALLOW", "reason": ""}]})
    numbers = [v for rule in policy["meme_mode"] + policy["text_mode"] for v in rule.values() if isinstance(v, float)]
    assert numbers == []                                                         # no confidence thresholds


# ---------------- error taxonomy ----------------

def _row(t, i, m, y=1, **flags):
    return {"true_label": y, "text_prediction": t, "image_prediction": i, "multimodal_prediction": m, **flags}


def test_error_taxonomy_flags_primary_category_and_error_type():
    allowed = {(c, s) for c, subs in er.TAXONOMY.items() for s in subs}
    rows = [_row(1, 1, 0), _row(1, 0, 0), _row(0, 0, 0, has_negation=True), _row(0, 1, 1, caption_empty=True),
            _row(0, 0, 0, low_resolution=True, has_watermark=True), _row(1, 1, 1, y=0)]
    for row in rows:
        assert set(er.auto_flags(row)) <= allowed
    assert er.primary_category(rows[0])["error_subcategory"] == "fusion failure"
    assert er.primary_category(rows[1])["error_subcategory"] == "followed the wrong modality"
    assert er.primary_category(rows[2])["error_subcategory"] == "negation"
    assert er.primary_category(rows[3])["error_subcategory"] == "missing text"
    assert er.primary_category(rows[4])["error_subcategory"] == "low resolution"
    manual = er.primary_category(rows[2], manual_label="cultural/contextual reference")
    assert manual == {"error_category": "CONTEXTUAL", "error_subcategory": "cultural reference", "category_source": "manual review"}
    assert all(v in allowed for v in er.MANUAL_MAP.values() if v is not None)
    fallback = er.primary_category(rows[0], manual_label="no clear content cause")
    assert fallback["error_subcategory"] == "fusion failure" and "no clear content cause" in fallback["category_source"]
    with pytest.raises(ValueError):
        er.primary_category(rows[0], manual_label="made-up category")
    assert er.error_type(rows[5], "multimodal") == "false_positive" and er.error_type(rows[1], "multimodal") == "false_negative"
    assert er.error_type(rows[3], "image") is None and er.error_type(rows[3], "text") == "false_negative"


# ---------------- config and the future UI object ----------------

def test_xai_config_matches_final_models_and_privacy_defaults():
    c = yaml.safe_load((REPO_ROOT / "config" / "explainability.yaml").read_text(encoding="utf-8"))
    finals = load_config()["final_models"]
    assert (c["models"]["text"], c["models"]["image"], c["models"]["multimodal"]) == (
        finals["FINAL_TEXT_MODEL"], finals["FINAL_IMAGE_MODEL"], finals["FINAL_MULTIMODAL_MODEL"])
    assert c["cache"]["cache_user_uploads"] is False and "model attribution" in c["ui_disclaimer"]
    xi.window_positions(224, c["image"]["window"], c["image"]["stride"])
    xi.window_positions(224, c["image"]["ui_window"], c["image"]["ui_stride"])


class _FakeReader:
    def __init__(self, output):
        self.output = output

    def readtext(self, array):
        return self.output


def _tiny_config():
    c = ex.load_xai_config()
    return {**c, "models": {"text": "FT", "image": "I01", "multimodal": "M02", "traditional_text": "LR"},
            "image": {**c["image"], "ui_window": 8, "ui_stride": 4}}


def test_explanation_object_meme_and_text_modes(registries):
    cfg = _tiny_config()
    img = synthetic_image((215, 25, 25))
    full = ex.explain_meme(img, caption="ladki kitchen me", config=cfg)
    assert set(full) >= {"prediction", "text_explanation", "image_explanation", "modality_comparison", "moderation",
                         "limitations", "disclaimer"} and full["stored"] is False
    assert full["modality_comparison"]["agreement"] in xm.AGREEMENT.values()
    assert full["moderation"]["recommendation"] in mod.OUTPUTS
    json.dumps(full)
    no_text = ex.explain_meme(img, reader=_FakeReader([]), config=cfg)              # OCR finds nothing -> image only
    assert no_text["ocr"]["route"] == "image_only" and no_text["text_explanation"] is None
    assert no_text["prediction"]["source"].startswith("image-only")
    bad = ex.explain_meme(b"garbage", reader=_FakeReader([]), config=cfg)
    assert bad["prediction"] is None and bad["moderation"]["recommendation"] is None
    txt = ex.explain_text_input("she is so dumb", config=cfg)
    assert txt["mode"] == "text" and txt["moderation"]["rule_id"] in ("T1", "T2", "T3") and txt["stored"] is False
    json.dumps(txt)
