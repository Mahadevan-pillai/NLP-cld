"""Tests for Task 14 OCR code. A FAKE reader replaces EasyOCR, so no OCR model is downloaded or run."""

import io

import numpy as np
import pytest
from PIL import Image

from src.ocr import cache as oc
from src.ocr import metrics as om
from src.ocr.engine import PREPARATIONS, extract_text_from_image, prepare
from src.ocr.pipeline import analyze_meme_input, load_ocr_config, preprocess_ocr_text, route
from src.preprocessing import PreprocessingConfig, preprocess

BOX = [[0, 0], [10, 0], [10, 5], [0, 5]]


class FakeReader:
    def __init__(self, output=None, error=None):
        self.output, self.error, self.calls = output or [], error, 0

    def readtext(self, array):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output


def png_bytes(size=(60, 40), color=(255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


POLICY = load_ocr_config()["fallback"]


# ---------------- engine wrapper ----------------

def test_extract_text_structure_unicode_and_confidence():
    reader = FakeReader([(BOX, "Ye ladki", 0.9), (BOX, "नहीं पता", 0.5)])
    r = extract_text_from_image(png_bytes(), reader=reader)
    assert r["status"] == "ok" and r["text"] == "Ye ladki नहीं पता" and r["confidence"] == 0.7
    assert len(r["detected_regions"]) == 2 and r["detected_regions"][1]["text"] == "नहीं पता"
    assert {"engine", "engine_version", "languages", "preparation", "pipeline_version", "error"} <= set(r)
    assert r["engine"] == "easyocr" and r["languages"] == ["en", "hi"]


def test_no_text_and_whitespace_only():
    assert extract_text_from_image(png_bytes(), reader=FakeReader([]))["status"] == "no_text"
    r = extract_text_from_image(png_bytes(), reader=FakeReader([(BOX, "   ", 0.4)]))
    assert r["status"] == "no_text" and r["text"] == "" and r["confidence"] is None


def test_invalid_corrupt_missing_images_do_not_crash(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    for source in (b"garbage", tmp_path / "bad.jpg", tmp_path / "missing.png", None):
        r = extract_text_from_image(source, reader=FakeReader([(BOX, "x", 1.0)]))
        assert r["status"] == "invalid_image" and r["text"] == "" and r["error"]


def test_engine_failure_is_reported_not_raised():
    r = extract_text_from_image(png_bytes(), reader=FakeReader(error=RuntimeError("boom")))
    assert r["status"] == "ocr_error" and "RuntimeError" in r["error"]


def test_preparation_variants_and_original_untouched(tmp_path):
    path = tmp_path / "m.png"
    Image.new("RGB", (30, 20), (100, 120, 140)).save(path)
    before = path.read_bytes()
    img = Image.open(path).convert("RGB")
    assert prepare(img, "none").shape == (20, 30, 3)
    gray = prepare(Image.fromarray(np.tile(np.array([[90], [110]], dtype=np.uint8), (10, 30))).convert("RGB"),
                   "grayscale_autocontrast")
    assert gray.ndim == 2 and gray.min() == 0 and gray.max() == 255          # contrast stretched
    with pytest.raises(ValueError):
        prepare(img, "sharpen")
    extract_text_from_image(path, preparation="grayscale_autocontrast", reader=FakeReader([]))
    assert path.read_bytes() == before and set(PREPARATIONS) == {"none", "grayscale_autocontrast"}


# ---------------- Task 6 preprocessing reuse ----------------

def test_ocr_text_uses_task6_preprocess_unchanged():
    text = "kr rha hu @someone https://x.co"
    out = preprocess_ocr_text(text)
    cfg = PreprocessingConfig()
    assert out == {"ocr_text_raw": preprocess(text, "raw", cfg), "ocr_text_basic": preprocess(text, "basic", cfg),
                   "ocr_text_hinglish_normalized": preprocess(text, "hinglish_normalized", cfg)}
    assert out["ocr_text_hinglish_normalized"] != out["ocr_text_raw"]


# ---------------- fallback policy and orchestration ----------------

def test_route_covers_every_policy_branch():
    ok = {"ocr_text_hinglish_normalized": "kuch text"}
    assert route({"status": "invalid_image", "confidence": None}, None, POLICY)["route"] == "reject"
    assert route({"status": "ocr_error", "confidence": None}, None, POLICY)["route"] == "image_only"
    assert route({"status": "no_text", "confidence": None}, {"ocr_text_hinglish_normalized": ""}, POLICY)["route"] == "image_only"
    assert route({"status": "ok", "confidence": 0.9}, {"ocr_text_hinglish_normalized": "  "}, POLICY)["route"] == "image_only"
    low = route({"status": "ok", "confidence": 0.1}, ok, POLICY)
    assert low["route"] == "image_only_and_human_review" and low["human_review"] is True
    assert route({"status": "ok", "confidence": 0.95}, ok, POLICY) == {"route": "multimodal", "reason": "OCR text available",
                                                                    "human_review": False}


def test_analyze_meme_input_returns_ocr_text_and_decision_without_storing():
    r = analyze_meme_input(png_bytes(), reader=FakeReader([(BOX, "Ladki kr rhi hai", 0.8)]))
    assert r["stored"] is False and r["ocr"]["status"] == "ok" and r["decision"]["route"] == "multimodal"
    assert set(r["text"]) == {"ocr_text_raw", "ocr_text_basic", "ocr_text_hinglish_normalized"}
    bad = analyze_meme_input(b"nope", reader=FakeReader([]))
    assert bad["text"] is None and bad["decision"]["route"] == "reject"


def test_ocr_config_policy_and_privacy_defaults():
    c = load_ocr_config()
    assert c["engine"] == "easyocr" and c["languages"] == ["en", "hi"]
    assert c["cache"]["cache_user_uploads"] is False and c["fallback"]["min_confidence"] == 0.30
    assert c["preparation"] in (None, *c["preparation_candidates"])


# ---------------- cache ----------------

def test_cache_roundtrip_key_and_failures_not_cached(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(png_bytes())
    cache = oc.OCRCache(tmp_path / "cache")
    reader = FakeReader([(BOX, "hello", 0.9)])
    run = lambda p, prep: extract_text_from_image(p, prep, reader=reader)      # noqa: E731
    first, hit1 = oc.cached_ocr(img, "s1", "none", cache, run)
    second, hit2 = oc.cached_ocr(img, "s1", "none", cache, run)
    assert (hit1, hit2) == (False, True) and first == second and reader.calls == 1
    _, hit3 = oc.cached_ocr(img, "s1", "grayscale_autocontrast", cache, run)  # different preparation -> new key
    assert hit3 is False and reader.calls == 2
    other = tmp_path / "b.png"
    other.write_bytes(png_bytes(color=(0, 0, 0)))                               # different image -> not cached yet
    failing = FakeReader(error=RuntimeError("x"))
    run_fail = lambda p, prep: extract_text_from_image(p, prep, reader=failing)   # noqa: E731
    oc.cached_ocr(other, "s2", "none", cache, run_fail)
    oc.cached_ocr(other, "s2", "none", cache, run_fail)
    assert failing.calls == 2                                                   # failures are retried, not cached
    k1 = oc.cache_key("h", "easyocr", "1.7.2", "none", "1.0")
    assert k1 != oc.cache_key("h", "easyocr", "1.7.3", "none", "1.0") and len(k1) == 64


# ---------------- CER / WER ----------------

def test_edit_distance_cer_wer():
    assert om.edit_distance("kitten", "sitting") == 3 and om.edit_distance(["a", "b"], ["b"]) == 1
    assert om.cer("abc", "abd") == pytest.approx(1 / 3) and om.cer("abc", "abc") == 0.0
    assert om.wer("a b c", "a x c") == pytest.approx(1 / 3) and om.wer("a b", "") == 1.0
    assert om.cer("Hello   World", "hello world") == 0.0                      # light normalisation
    assert om.cer("", "anything") is None and om.wer("  ", "x") is None          # empty reference: undefined
    assert om.cer("नहीं", "नही") == pytest.approx(1 / 4)                         # Unicode code points
    c = om.corpus_rates([("ab", "ab"), ("abcd", "abxd"), ("", "zzz")])
    assert c["cer"] == pytest.approx(1 / 6) and c["reference_chars"] == 6
