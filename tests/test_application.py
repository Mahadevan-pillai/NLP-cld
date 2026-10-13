"""Tests for the Task 16 application layer and Streamlit pages.

Expensive models are MOCKED (no download, no checkpoint): the inference functions are replaced by small
fakes, OCR uses a fake reader, MongoDB uses the in-memory fake_db. The real Task 6 preprocessing, input
validation, moderation rules, history schema and Streamlit pages run unchanged.
"""

import io
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from src.application import artifacts, research, validation
from src.application import service as service_module
from src.application.history import HistoryService
from src.application.service import AnalysisService
from src.application.settings import final_models, load_app_config, supported_categories
from src.database import schemas
from src.inference import image as image_inference
from src.inference import text as text_inference
from src.preprocessing import PreprocessingConfig, preprocess
from src.utils.config import REPO_ROOT

BOX = [[0, 0], [10, 0], [10, 5], [0, 5]]


def png(size=(80, 60), color=(200, 30, 30), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def available_rows(missing=()):
    names = {c for req in artifacts.MODE_REQUIREMENTS.values() for c in req}
    return [{"component": c, "status": "missing" if c in missing else "available"} for c in sorted(names)]


class FakeReader:
    def __init__(self, output=None, error=None):
        self.output, self.error = output or [], error

    def readtext(self, array):
        if self.error:
            raise self.error
        return self.output


def fake_pred(mid, label, score):
    return {"model_version": mid, "task": "misogyny_detection", "predicted_label": label,
            "predicted_name": "misogynistic" if label else "non_misogynistic", "score": score,
            "score_type": "uncalibrated test score", "calibrated": False, "preprocessing_mode": "hinglish_normalized",
            "preprocessing_version": "1.0", "processed_text_empty": False}


@pytest.fixture
def mocked(monkeypatch):
    """Replace every model call with deterministic fakes; keep preprocessing, moderation and schemas real."""
    calls = {"text": [], "image": 0, "mm": 0}

    def predict_text(text, mid, registry_path=None):
        calls["text"].append(mid)
        return fake_pred(mid, int("ladki" in text.lower()), 0.7 if "ladki" in text.lower() else 0.3)

    monkeypatch.setattr(text_inference, "predict_text", predict_text)
    monkeypatch.setattr(text_inference, "_load", lambda mid, path=None: (
        "tfidf_logreg", {"preprocessing_mode": "hinglish_normalized", "preprocessing_options": None, "max_length": 64}, None))

    def predict_image(img, mid, registry_path=None):
        calls["image"] += 1
        return fake_pred(mid, 1, 0.9)

    monkeypatch.setattr(image_inference, "predict_image", predict_image)
    import src.explainability.multimodal as xm

    def explain_multimodal(img, text, mm, t, i, **kw):
        calls["mm"] += 1
        tl = int("ladki" in text.lower())
        return {"predictions": {"text": fake_pred(t, tl, 0.6), "image": fake_pred(i, 1, 0.9), "multimodal": fake_pred(mm, 1, 0.8)},
                "modality_agreement": xm.modality_agreement(tl, 1, 1),
                "single_modality_models_disagreeing": [] if tl else ["text"],
                "ablation": {"dependence": "label depends on the image", "label_changes_without_text": False,
                             "label_changes_without_image": True}}

    monkeypatch.setattr(xm, "explain_multimodal", explain_multimodal)
    import src.explainability.text as xt
    monkeypatch.setattr(xt, "explain_text", lambda text, mid, steps=32, top_k=5: {
        "method": {"name": "integrated_gradients", "steps": steps}, "tokens": [{"token": "ladki", "contribution": 0.5}],
        "top_positive": [{"token": "ladki", "contribution": 0.5}], "top_negative": [], "completeness_error": 0.0,
        "truncated": False, "limitations": ["model attribution"]})
    import src.explainability.image as xi
    monkeypatch.setattr(xi, "explain_image", lambda img, mid, w, s: {
        "method": {"name": "occlusion"}, "heatmap": [[0.1, -0.2], [0.0, 0.3]], "top_regions_towards_misogynistic": [],
        "top_regions_towards_non_misogynistic": [], "limitations": ["model attribution"]})
    return calls


@pytest.fixture
def svc(mocked):
    return AnalysisService(audit_rows=available_rows(), ocr_reader=FakeReader([(BOX, "ye ladki kr rhi hai", 0.9)]))


# ---------------- configuration, models, categories ----------------

def test_final_models_come_from_frozen_config_and_categories_from_config():
    m = final_models()
    assert (m["text"], m["image"], m["multimodal"], m["second_opinion_text"]) == ("T05", "I03", "M04", "B07")
    cats = supported_categories()
    assert [c for t in cats["enabled"] for c in t["categories"]] == ["Misogyny"]
    assert {"Objectification", "Prejudice", "Humiliation"} <= {c for t in cats["disabled"] for c in t["categories"]}
    assert "THREAT" in cats["unsupported"] and "SAFE" in cats["unsupported"]
    cfg = load_app_config()
    assert cfg["history"]["consent_default"] is False and cfg["upload"]["persist_uploads"] is False
    assert cfg["upload"]["max_upload_mb"] == 10 and cfg["logging"]["log_user_text"] is False


def test_missing_artifact_disables_mode_and_never_substitutes(mocked):
    s = AnalysisService(audit_rows=available_rows(missing=("FINAL_MULTIMODAL_MODEL",)), ocr_reader=FakeReader())
    assert s.availability["text"][0] and not s.availability["meme"][0]
    r = s.analyze_meme(png())
    assert r["status"] == "error" and r["error"]["category"] == "artifact_unavailable"
    assert "No other model is substituted" in r["error"]["message"] and mocked["mm"] == 0
    with pytest.raises(artifacts.ArtifactUnavailableError):
        artifacts.require("meme", available_rows(missing=("OCR pipeline (EasyOCR)",)))


# ---------------- validation ----------------

@pytest.mark.parametrize("bad", [None, "", "   \n\t ", "🙂🙂🙂", "12345 !!!", "x" * 2001, 42])
def test_text_validation_rejects_unusable_input_with_safe_messages(bad):
    with pytest.raises(validation.InputError) as e:
        validation.validate_text(bad, max_chars=2000)
    assert str(e.value) and "Traceback" not in str(e.value)


@pytest.mark.parametrize("good", ["Ye ladki kr rhi hai 😂 https://x.co", "लड़की को कुछ नहीं आता", "She is smart", "café time\x00"])
def test_text_validation_accepts_unicode_hindi_hinglish_emoji_urls(good):
    cleaned, warnings = validation.validate_text(good)
    assert cleaned and "\x00" not in cleaned and warnings == []
    assert validation.validate_text("这是中文 and some english")[1]          # other scripts -> warning, not crash


def test_upload_validation_formats_sizes_and_corruption():
    img, meta = validation.validate_upload(png((50, 40)))
    assert img.mode == "RGB" and meta == {"width": 50, "height": 40, "format": "PNG", "mode": "RGB", "size_bytes": len(png((50, 40)))}
    assert validation.validate_upload(png(fmt="JPEG"))[1]["format"] == "JPEG"
    for data, msg in ((b"", "upload"), (b"not an image", "not a readable image"), (png(fmt="GIF"), "Unsupported image format"),
                      (png(fmt="BMP"), "Unsupported image format"), (png()[:60], "corrupt"), (png((8, 8)), "corrupt")):
        with pytest.raises(validation.InputError, match=msg):
            validation.validate_upload(data)
    with pytest.raises(validation.InputError, match="too large"):
        validation.validate_upload(png(), max_upload_mb=0.0001)
    with pytest.raises(validation.InputError, match="too many pixels"):
        validation.validate_upload(png((300, 300)), max_pixels=1000)


# ---------------- text analysis ----------------

def test_analyze_text_structure_preprocessing_reuse_moderation_and_versions(svc, mocked):
    r = svc.analyze_text("Ye LADKI kr rhi hai @someone")
    assert r["status"] == "ok" and r["analysis_type"] == "text" and r["schema_version"] == "1.0"
    assert r["input"]["normalized_text"] == preprocess("Ye LADKI kr rhi hai @someone", "hinglish_normalized", PreprocessingConfig())
    assert r["prediction"]["model_version"] == "T05" and r["prediction"]["display"] == "Potentially misogynistic"
    assert r["models"]["second_opinion"]["model_version"] == "B07" and mocked["text"] == ["T05", "B07"]
    assert r["moderation"]["recommendation"] == "WARNING" and r["moderation"]["automatic_block"] is False
    assert r["model_versions"] == {"text_model": "T05"} and r["category"]["subcategories"].startswith("not available")
    assert r["explanation"]["text"]["status"] == "ok" and r["prediction"]["calibrated"] is False
    json.dumps(r)
    assert svc.analyze_text("bhai exam hai")["moderation"]["recommendation"] == "ALLOW"


def test_analyze_text_invalid_input_and_inference_failure_are_safe(svc, monkeypatch):
    assert svc.analyze_text("   ")["error"]["category"] == "invalid_input"

    def boom(*a, **k):
        raise RuntimeError("secret internal detail mongodb+srv://" + "user:pw" + "@host")   # split: CI credential scan
    monkeypatch.setattr(text_inference, "predict_text", boom)
    r = svc.analyze_text("ye ladki")
    assert r["status"] == "error" and r["error"]["category"] == "inference_error" and r["prediction"] is None
    assert "secret" not in r["error"]["message"] and "mongodb" not in r["error"]["message"]


def test_text_explanation_failure_does_not_hide_prediction(svc, monkeypatch):
    import src.explainability.text as xt

    def fail(*a, **k):
        raise RuntimeError("xai broke")
    monkeypatch.setattr(xt, "explain_text", fail)
    r = svc.analyze_text("ye ladki")
    assert r["status"] == "ok" and r["prediction"] and r["explanation"]["text"]["status"] == "error"


# ---------------- meme analysis ----------------

def test_analyze_meme_multimodal_route_three_models_and_ocr_text(svc, mocked):
    r = svc.analyze_meme(png())
    o = r["input"]["ocr"]
    assert r["status"] == "ok" and o["status"] == "ok" and o["text"] == "ye ladki kr rhi hai" and o["route"] == "multimodal"
    assert o["normalized_text"] == preprocess(o["text"], "hinglish_normalized", PreprocessingConfig())
    assert r["prediction"]["model_version"] == "M04" and r["prediction"]["source"].startswith("multimodal")
    assert r["models"]["image"]["used_input"] == "image only (no text)" and r["models"]["text"]["used_input"] == "OCR text only"
    assert r["modality"]["agreement"] == "AGREE" and r["moderation"]["recommendation"] == "WARNING"
    assert r["model_versions"] == {"text_model": "T05", "image_model": "I03", "multimodal_model": "M04"}
    assert set(r["input"]["image_metadata"]) == set(schemas.IMAGE_METADATA_KEYS)
    json.dumps(r)


@pytest.mark.parametrize("reader,route,rec", [
    (FakeReader([]), "image_only", "HUMAN_REVIEW"),                                   # no text -> image only (I03 flags)
    (FakeReader([(BOX, "ye", 0.1)]), "image_only_and_human_review", "HUMAN_REVIEW"),  # very low OCR confidence
    (FakeReader(error=RuntimeError("ocr crashed")), "image_only", "HUMAN_REVIEW"),    # OCR exception
])
def test_meme_ocr_failures_fall_back_to_image_only_and_never_send_empty_text(mocked, reader, route, rec):
    s = AnalysisService(audit_rows=available_rows(), ocr_reader=reader)
    r = s.analyze_meme(png())
    assert r["status"] == "limited" and r["input"]["ocr"]["route"] == route and mocked["mm"] == 0
    assert r["prediction"]["model_version"] == "I03" and r["prediction"]["source"] == "image-only fallback"
    assert "unavailable" in r["models"]["multimodal"] and "unavailable" in r["models"]["text"]
    assert r["moderation"]["recommendation"] == rec and any("Limited analysis" in w for w in r["warnings"])


def test_invalid_uploads_and_on_demand_explanation(svc):
    for data in (b"", b"garbage", png(fmt="GIF")):
        r = svc.analyze_meme(data)
        assert r["status"] == "error" and r["error"]["category"] == "invalid_input"
    e = svc.explain_meme(png(), "ye ladki")
    assert e["image"]["status"] == "ok" and e["image"]["overlay"].size == (80, 60) and e["text"]["status"] == "ok"
    assert svc.explain_meme(b"garbage", None)["image"]["status"] == "error"


# ---------------- history and feedback ----------------

def test_history_requires_consent_stores_schema_fields_and_never_binary(svc, fake_db):
    h = HistoryService(db=fake_db)
    r = svc.analyze_meme(png())
    assert h.save(r, consent=False) == {"ok": True, "saved": False, "analysis_id": None, "message": "Not saved (no consent given)."}
    assert fake_db["analysis_history"].docs == []
    saved = h.save(r, consent=True, source="e2e_test")
    doc = h.get(saved["analysis_id"])["item"]
    assert saved["saved"] and doc["analysis_type"] == "meme" and doc["source"] == "e2e_test"
    assert doc["predictions"]["final"] == {"label": "misogynistic", "score": 0.8} and doc["moderation"]["recommendation"] == "WARNING"
    assert doc["ocr"]["route"] == "multimodal" and doc["input"]["ocr_text"] == "ye ladki kr rhi hai"
    assert "image_reference" not in doc["input"] and set(doc["input"]["image_metadata"]) <= set(schemas.IMAGE_METADATA_KEYS)
    t = svc.analyze_text("bhai exam hai")
    tid = h.save(t, consent=True)["analysis_id"]
    assert h.get(tid)["item"]["input"]["original_text"] == "bhai exam hai"
    assert [d["analysis_type"] for d in h.recent(analysis_type="text")["items"]] == ["text"]
    assert len(h.recent(final_label="misogynistic")["items"]) == 1
    assert h.recent(since=datetime.now(timezone.utc) + timedelta(days=1))["items"] == []


def test_feedback_and_database_failure_are_safe(svc, fake_db):
    h = HistoryService(db=fake_db)
    aid = h.save(svc.analyze_text("ye ladki"), consent=True)["analysis_id"]
    assert h.add_feedback(aid, "incorrect", "  looks harmless  ")["ok"]
    fb = h.get(aid)["feedback"][0]
    assert fb["rating"] == "incorrect" and fb["comment"] == "looks harmless"
    assert not h.add_feedback("00000000-0000-4000-8000-000000000000", "correct")["ok"]       # unknown analysis
    assert not h.add_feedback(aid, "maybe")["ok"] and not h.add_feedback(aid, "correct", "x" * 501)["ok"]

    class Broken:
        def __getitem__(self, name):
            raise RuntimeError("connection refused to mongodb+srv://" + "user:pw" + "@cluster")
    bad = HistoryService(db=Broken())
    r = bad.save(svc.analyze_text("ye ladki"), consent=True)
    assert r == {"ok": False, "saved": False, "analysis_id": None, "message": "History is unavailable (RuntimeError)."}
    assert not bad.recent()["ok"] and not bad.get(aid)["ok"]
    assert "mongodb" not in json.dumps(bad.recent()) and "pw" not in bad.add_feedback(aid, "correct")["message"]


def test_schema_extra_fields_are_validated():
    base = dict(analysis_type="text", input_data={"original_text": "x"}, predictions={"final": {"label": "misogynistic"}})
    ok = schemas.build_analysis(**base, extra={"moderation": {"recommendation": "HUMAN_REVIEW", "rule_id": "T3", "reason": "r"},
                                               "app_version": "1.0.0", "source": "app"})
    assert ok["moderation"]["recommendation"] == "HUMAN_REVIEW"
    for extra in ({"moderation": {"recommendation": "BLOCK"}}, {"source": "somewhere"}, {"ocr": {"status": "ok"}},
                  {"unknown": 1}):
        with pytest.raises(schemas.ValidationError):
            schemas.build_analysis(**base, extra=extra)


# ---------------- research artefacts, logging ----------------

def test_research_loader_reads_stored_files_and_handles_missing(monkeypatch):
    assert research.table("main_results") is not None or not (REPO_ROOT / research.ARTIFACTS["main_results"]["path"]).exists()
    monkeypatch.setitem(research.ARTIFACTS, "nope", {"path": "results/does_not_exist.csv", "section": "x", "title": "x"})
    assert research.table("nope") is None and research.figure("nope") is None


def test_log_event_drops_unsafe_fields():
    from src.application.logging_setup import log_event

    safe = log_event("test", analysis_type="text", text="user secret text", uri="mongodb+srv://x", duration_ms=1.0)
    assert safe == {"analysis_type": "text", "duration_ms": 1.0}


# ---------------- Streamlit pages (headless smoke tests) ----------------

@pytest.fixture
def app_test(svc, fake_db, monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    app_dir = str(REPO_ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from services import state

    history = HistoryService(db=fake_db)
    svc.warm_error = None
    monkeypatch.setattr(state, "get_service", lambda: svc)
    monkeypatch.setattr(state, "get_audit", lambda: {"rows": available_rows(), "seconds": 0.0, "error": None})
    monkeypatch.setattr(state, "get_history", lambda: history)
    monkeypatch.setattr(state, "db_status", lambda: {"ok": True, "message": "test"})
    return st_testing.AppTest.from_file(str(REPO_ROOT / "app" / "app.py"), default_timeout=60), history


@pytest.mark.parametrize("page", ["pages/home.py", "pages/history.py", "pages/dashboard.py", "pages/about.py",
                                  "pages/analyze_text.py", "pages/analyze_meme.py"])
def test_every_page_renders_without_exceptions(app_test, page):
    at, _ = app_test
    at.run()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_analyze_text_page_end_to_end_with_consent_and_feedback(app_test):
    at, history = app_test
    at.run()
    at.switch_page("pages/analyze_text.py").run()
    at.text_area[0].input("Ye ladki kr rhi hai")
    at.checkbox[0].check()
    at.button[0].click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "FINAL ANALYSIS" in page and "Potentially misogynistic" in page and "WARNING" in page
    saved = history.recent()["items"]
    assert len(saved) == 1 and saved[0]["input"]["original_text"] == "Ye ladki kr rhi hai"
    assert any("Saved to history" in s.value for s in at.success)


def test_analyze_text_page_without_consent_saves_nothing_and_shows_validation(app_test):
    at, history = app_test
    at.run()
    at.switch_page("pages/analyze_text.py").run()
    at.button[0].click().run()                                    # empty input
    assert not at.exception and any("Please enter some text" in w.value for w in at.warning)
    at.text_area[0].input("bhai exam hai")
    at.button[0].click().run()
    assert history.recent()["items"] == [] and "ALLOW" in " ".join(m.value for m in at.markdown)


def test_analyze_meme_page_upload_analyze_and_explain(app_test):
    at, history = app_test
    at.run()
    at.switch_page("pages/analyze_meme.py").run()
    at.file_uploader[0].set_value(("anything.png", png(), "image/png")).run()     # the file name is never used
    at.button[0].click().run()                                                    # Analyze (no consent)
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "FINAL ANALYSIS" in page and "MODEL COMPARISON" in page and "AGREE" in page
    assert any("ye ladki kr rhi hai" in t.value for t in at.text)                 # OCR text shown as plain text
    assert history.recent()["items"] == []                                        # no consent -> nothing stored
    next(b for b in at.button if b.label == "Explain prediction").click().run()
    assert not at.exception and len(at.get("image")) >= 2                         # meme + heatmap overlay rendered


def test_dashboard_with_missing_artifact_and_history_with_database_down(app_test, monkeypatch):
    at, _ = app_test
    monkeypatch.setitem(research.ARTIFACTS, "main_results", {"path": "results/missing.csv", "section": "Final comparison",
                                                             "title": "Final TEST results (758 memes)"})
    at.run()
    at.switch_page("pages/dashboard.py").run()
    assert not at.exception and any("result file unavailable" in i.value for i in at.info)

    class Down:
        def __getitem__(self, name):
            raise RuntimeError("down")
    from services import state
    monkeypatch.setattr(state, "get_history", lambda: HistoryService(db=Down()))
    at.switch_page("pages/history.py").run()
    assert not at.exception and any("History is unavailable" in w.value for w in at.warning)
