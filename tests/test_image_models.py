"""Tests for Task 10 image code using synthetic images and a TINY randomly initialised SigLIP.

No pretrained model is downloaded and no dataset image is needed.
"""

import io
import json

import joblib
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import SiglipImageProcessorPil, SiglipVisionConfig, SiglipVisionModel

from src.data.eda import class_names
from src.inference import image as inference
from src.models import image_encoder as ie
from src.preprocessing.image import IMAGE_PREPROCESSING_VERSION, ImageLoadError, load_image
from src.utils.config import REPO_ROOT, load_config


def synthetic(color, size=(48, 40), mode="RGB"):
    return Image.new(mode, size, color)


def jpeg_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def tiny():
    processor = SiglipImageProcessorPil(size={"height": 32, "width": 32}, resample=3, do_rescale=True,
                                        rescale_factor=1 / 255, do_normalize=True,
                                        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    torch.manual_seed(0)
    cfg = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                             image_size=32, patch_size=8)
    model = SiglipVisionModel(cfg).eval()
    return processor, model


# Mixed images: classes differ by colour, so a linear classifier can learn them.
IMAGES = [synthetic((200 + i, 10, 10)) for i in range(10)] + [synthetic((10, 10, 200 + i)) for i in range(10)]
Y = np.array([1] * 10 + [0] * 10)


# ---------------- image loading ----------------

def test_load_image_sources_and_rgb(tmp_path):
    path = tmp_path / "a.png"
    synthetic((1, 2, 3, 128), mode="RGBA").save(path)
    for source in (path, str(path), path.read_bytes(), Image.open(path)):
        img = load_image(source)
        assert img.mode == "RGB" and img.size == (48, 40)
    assert load_image(synthetic(77, mode="L")).getpixel((0, 0)) == (77, 77, 77)            # grayscale -> RGB
    assert load_image(synthetic(3, mode="P")).mode == "RGB"                                  # palette -> RGB


def test_load_image_applies_exif_orientation():
    img = synthetic((9, 9, 9), size=(60, 30))
    exif = img.getexif()
    exif[274] = 6                                              # "rotate 90 CW" orientation tag
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    assert load_image(buf.getvalue()).size == (30, 60)


def test_load_image_rejects_bad_input(tmp_path):
    good = jpeg_bytes(synthetic((5, 5, 5), size=(200, 200)))
    (tmp_path / "corrupt.jpg").write_bytes(b"not an image at all")
    (tmp_path / "truncated.jpg").write_bytes(good[: len(good) // 3])
    bad = [None, 12345, tmp_path / "missing.jpg", tmp_path / "corrupt.jpg", tmp_path / "truncated.jpg",
           b"", synthetic((1, 1, 1), size=(8, 300))]
    for source in bad:
        with pytest.raises(ImageLoadError):
            load_image(source)


def test_load_image_never_modifies_file(tmp_path):
    path = tmp_path / "m.jpg"
    path.write_bytes(jpeg_bytes(synthetic((1, 50, 9))))
    before = path.read_bytes()
    load_image(path)
    assert path.read_bytes() == before


# ---------------- preprocessing and encoder ----------------

def test_processor_stretches_to_square_and_normalises(tiny):
    processor, _ = tiny
    pv = processor(images=[load_image(synthetic((255, 0, 0), size=(90, 30)))], return_tensors="pt")["pixel_values"]
    assert tuple(pv.shape) == (1, 3, 32, 32)                        # no crop: whole image stretched
    assert float(pv.max()) <= 1.0 + 1e-6 and float(pv.min()) >= -1.0 - 1e-6     # mean/std 0.5 -> [-1, 1]
    np.testing.assert_allclose(pv[0, 0].numpy(), 1.0, atol=1e-5)  # pure red channel
    np.testing.assert_allclose(pv[0, 2].numpy(), -1.0, atol=1e-5)


def test_embeddings_shape_order_batch_invariance_and_frozen(tiny):
    processor, model = tiny
    before = {k: v.clone() for k, v in model.state_dict().items()}
    batched = ie.image_embeddings(IMAGES, processor, model, batch_size=7)
    single = np.concatenate([ie.image_embeddings([im], processor, model, batch_size=1) for im in IMAGES])
    assert batched.shape == (20, 32) and batched.dtype == np.float32
    np.testing.assert_allclose(batched, single, atol=1e-5)          # same sample order, batch size irrelevant
    np.testing.assert_allclose(batched, ie.image_embeddings(IMAGES, processor, model), atol=1e-6)   # deterministic
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())                    # frozen


def test_embedding_cache_and_stale_detection(tiny, tmp_path):
    processor, model = tiny
    ids, shas = [f"s{i}" for i in range(4)], [f"h{i}" for i in range(4)]
    meta = ie.cache_metadata("train", ids, shas, "tiny/siglip", "rev1")
    emb, _, cached = ie.embed_cached(IMAGES[:4], meta, tmp_path, processor, model, log=lambda *_: None)
    assert not cached and emb.shape == (4, 32)
    again, _, cached = ie.embed_cached(IMAGES[:4], meta, tmp_path, None, None)
    assert cached and np.array_equal(emb, again)
    stale = [ie.cache_metadata("train", ids[::-1], shas[::-1], "tiny/siglip", "rev1"),     # different order
             ie.cache_metadata("train", ids, ["x"] + shas[1:], "tiny/siglip", "rev1"),     # an image changed
             ie.cache_metadata("train", ids, shas, "tiny/siglip", "rev2"),                 # different encoder
             ie.cache_metadata("validation", ids, shas, "tiny/siglip", "rev1")]            # different split file
    for m in stale:
        assert not ie.cache_is_valid(tmp_path / "train.npy", tmp_path / "train.json", m)


def test_load_cached_embeddings_checks_order(tiny, tmp_path, monkeypatch):
    processor, model = tiny
    monkeypatch.setattr(ie, "embedding_cache_dir", lambda config: tmp_path)
    meta = ie.cache_metadata("validation", ["a", "b"], ["1", "2"], "tiny", "r")
    ie.embed_cached(IMAGES[:2], meta, tmp_path, processor, model, log=lambda *_: None)
    assert ie.load_cached_embeddings({}, "validation", ["a", "b"]).shape == (2, 32)
    with pytest.raises(RuntimeError):
        ie.load_cached_embeddings({}, "validation", ["b", "a"])


# ---------------- classifiers ----------------

@pytest.fixture(scope="module")
def embeddings(tiny):
    processor, model = tiny
    return ie.image_embeddings(IMAGES, processor, model)


@pytest.mark.parametrize("kind", ["logistic_regression", "linear_svm"])
def test_classifier_shapes_scores_and_reload(embeddings, kind, tmp_path):
    settings = load_config()["image_model"][kind]
    clf = ie.build_classifier(kind, settings, 42).fit(embeddings, Y)
    pred, scores = clf.predict(embeddings), ie.classifier_scores(clf, embeddings)
    assert pred.shape == (20,) and set(pred) <= {0, 1} and scores.shape == (20,)
    if kind == "logistic_regression":
        assert ((scores >= 0) & (scores <= 1)).all()
    joblib.dump(clf, tmp_path / "c.joblib")
    again = joblib.load(tmp_path / "c.joblib")
    assert np.array_equal(again.predict(embeddings), pred) and np.allclose(ie.classifier_scores(again, embeddings), scores)


def test_build_classifier_rejects_unknown():
    with pytest.raises(ValueError):
        ie.build_classifier("mlp", {}, 0)


# ---------------- inference interface ----------------

@pytest.fixture
def registry(tiny, embeddings, tmp_path, monkeypatch):
    processor, model = tiny
    monkeypatch.setattr(inference, "_encoder", lambda checkpoint, revision: (processor, model))
    monkeypatch.setattr(inference, "REPO_ROOT", tmp_path)
    inference._load.cache_clear()
    models = {}
    for exp, kind, reg_kind in (("I01", "logistic_regression", "frozen_logreg"), ("I02", "linear_svm", "frozen_svm")):
        clf = ie.build_classifier(kind, load_config()["image_model"][kind], 42).fit(embeddings, Y)
        joblib.dump(clf, tmp_path / f"{exp}.joblib")
        models[exp] = {"kind": reg_kind, "artifact": f"{exp}.joblib", "encoder": "tiny/siglip", "encoder_revision": "r" * 40,
                       "image_preprocessing_version": IMAGE_PREPROCESSING_VERSION}
    models["OLD"] = {**models["I01"], "image_preprocessing_version": "0.9"}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    yield path
    inference._load.cache_clear()


@pytest.mark.parametrize("version", ["I01", "I02"])
def test_predict_image_structure_and_honest_scores(registry, version):
    out = inference.predict_image(jpeg_bytes(synthetic((230, 10, 10))), version, registry_path=registry)
    assert out["predicted_label"] in (0, 1) and out["predicted_name"] in ("non_misogynistic", "misogynistic")
    assert out["modality"] == "image" and out["text_used"] is False and out["calibrated"] is False
    assert out["model_version"] == version and out["image_preprocessing_version"] == IMAGE_PREPROCESSING_VERSION
    if version == "I01":
        assert "uncalibrated" in out["score_type"] and 0.0 <= out["score"] <= 1.0
    else:
        assert "NOT a probability" in out["score_type"]


def test_predict_image_matches_batch_pipeline_and_is_deterministic(registry, tiny, embeddings):
    clf = joblib.load(registry.parent / "I01.joblib")
    expected = clf.predict(embeddings[:1])[0]
    a = inference.predict_image(IMAGES[0], "I01", registry_path=registry)
    b = inference.predict_image(IMAGES[0], "I01", registry_path=registry)
    assert a == b and a["predicted_label"] == expected


def test_predict_image_rejects_bad_input_and_version_mismatch(registry):
    with pytest.raises(ImageLoadError):
        inference.predict_image(b"garbage", "I01", registry_path=registry)
    with pytest.raises(RuntimeError):
        inference.predict_image(IMAGES[0], "OLD", registry_path=registry)


# ---------------- labels, config, test-set protection, no text features ----------------

def test_label_mapping_matches_config():
    assert ie.LABEL_NAMES == class_names(load_config())


def test_image_config_is_pinned_untuned_baseline():
    icfg = load_config()["image_model"]
    assert icfg["encoder"]["checkpoint"] == "google/siglip-base-patch16-224"
    assert len(icfg["encoder"]["revision"]) == 40 and icfg["encoder"]["embedding_dim"] == 768
    assert icfg["experiments"] == {"I01": "logistic_regression", "I02": "linear_svm"}
    assert icfg["preprocessing"]["loader_version"] == IMAGE_PREPROCESSING_VERSION
    assert icfg["preprocessing"]["processor_backend"] == "pil"


TASK10_SOURCES = ("scripts/train_image_models.py", "scripts/report_image_models.py", "src/models/image_encoder.py",
                  "src/inference/image.py", "src/preprocessing/image.py")


def test_task10_sources_never_touch_test_split():
    for rel in TASK10_SOURCES:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'load_split("test"' not in source and "allow_test=True" not in source, rel


def test_image_model_code_takes_no_text_input():
    for rel in ("src/models/image_encoder.py", "src/inference/image.py", "src/preprocessing/image.py"):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for column in ("text_raw", "text_basic", "text_hinglish_normalized", "original_text", "image_reference"):
            assert column not in source, (rel, column)
