"""Tests for document construction and validation (src/database/schemas.py)."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.database import schemas
from src.database.schemas import ValidationError


def text_analysis(**overrides):
    kwargs = dict(analysis_type="text", input_data={"original_text": "example input"})
    kwargs.update(overrides)
    return schemas.build_analysis(**kwargs)


# ---------------- IDs and timestamps ----------------

def test_new_id_is_unique_uuid4():
    a, b = schemas.new_id(), schemas.new_id()
    assert a != b
    assert uuid.UUID(a).version == 4


def test_utc_now_is_timezone_aware_utc_millisecond_precision():
    now = schemas.utc_now()
    assert now.tzinfo is not None and now.utcoffset() == timedelta(0)
    assert now.microsecond % 1000 == 0


# ---------------- analysis_history ----------------

def test_build_text_analysis():
    doc = text_analysis(
        input_data={"original_text": "raw", "normalized_text": "norm"},
        predictions={"text_model": {"label": "some_label", "score": 0.8}, "final": {"label": "some_label"}},
        model_versions={"text_model": schemas.new_id()},
        processing_time_ms=12.5,
    )
    uuid.UUID(doc["analysis_id"])
    assert doc["analysis_type"] == "text"
    assert doc["created_at"].tzinfo is not None


def test_build_meme_analysis_all_prediction_slots():
    doc = schemas.build_analysis(
        "meme",
        {"image_reference": "tmp/abc", "image_metadata": {"width": 10, "height": 20, "format": "PNG"},
         "ocr_text": "ocr", "normalized_text": "norm"},
        predictions={k: {"label": "x", "score": -1.3} for k in schemas.PREDICTION_KEYS["meme"]},
    )
    assert doc["analysis_type"] == "meme"


def test_meme_analysis_fields_all_optional():
    # e.g. image kept temporary only and OCR found no text
    schemas.build_analysis("meme", {})


@pytest.mark.parametrize("bad_type", ["image", "", None, "TEXT"])
def test_invalid_analysis_type(bad_type):
    with pytest.raises(ValidationError):
        schemas.build_analysis(bad_type, {"original_text": "x"})


def test_text_analysis_requires_original_text():
    with pytest.raises(ValidationError):
        text_analysis(input_data={"original_text": "   "})


def test_text_analysis_rejects_meme_fields():
    with pytest.raises(ValidationError):
        text_analysis(input_data={"original_text": "x", "ocr_text": "y"})


def test_meme_analysis_rejects_original_text():
    with pytest.raises(ValidationError):
        schemas.build_analysis("meme", {"original_text": "x"})


def test_text_analysis_rejects_image_predictions():
    with pytest.raises(ValidationError):
        text_analysis(predictions={"image_model": {"label": "x"}})


@pytest.mark.parametrize("value", [-1, -0.1, True, "10", float("nan"), float("inf")])
def test_invalid_processing_time(value):
    with pytest.raises(ValidationError):
        text_analysis(processing_time_ms=value)


def test_zero_processing_time_allowed():
    text_analysis(processing_time_ms=0)


def test_rejects_binary_image_data():
    with pytest.raises(ValidationError, match="binary"):
        schemas.build_analysis("meme", {"image_reference": "ref", "image_metadata": {"width": b"\x89PNG"}})


def test_rejects_embedded_data_uri():
    with pytest.raises(ValidationError):
        schemas.build_analysis("meme", {"image_reference": "data:image/png;base64,AAAA"})


def test_rejects_non_allowlisted_image_metadata():
    with pytest.raises(ValidationError):
        schemas.build_analysis("meme", {"image_metadata": {"exif": {"GPSInfo": "..."}}})


def test_rejects_unknown_top_level_field():
    doc = text_analysis()
    doc["unexpected"] = 1
    with pytest.raises(ValidationError):
        schemas.validate_analysis(doc)


def test_prediction_requires_label():
    with pytest.raises(ValidationError):
        text_analysis(predictions={"final": {"score": 0.5}})


def test_prediction_rejects_non_finite_score():
    with pytest.raises(ValidationError):
        text_analysis(predictions={"final": {"label": "x", "score": float("nan")}})


def test_labels_unrestricted_by_default_but_configurable():
    preds = {"final": {"label": "label_a"}}
    text_analysis(predictions=preds)  # no restriction when allowed_labels is None
    text_analysis(predictions=preds, allowed_labels=["label_a", "label_b"])
    with pytest.raises(ValidationError):
        text_analysis(predictions=preds, allowed_labels=["label_b"])


def test_model_versions_slots_validated():
    with pytest.raises(ValidationError):
        text_analysis(model_versions={"final": "v1"})


def test_rejects_invalid_analysis_id():
    doc = text_analysis()
    doc["analysis_id"] = "not-a-uuid"
    with pytest.raises(ValidationError):
        schemas.validate_analysis(doc)


def test_rejects_naive_timestamp():
    doc = text_analysis()
    doc["created_at"] = datetime.now()
    with pytest.raises(ValidationError, match="timezone-aware"):
        schemas.validate_analysis(doc)


def test_rejects_non_utc_timestamp():
    doc = text_analysis()
    doc["created_at"] = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    with pytest.raises(ValidationError, match="UTC"):
        schemas.validate_analysis(doc)


# ---------------- user_feedback ----------------

@pytest.mark.parametrize("rating", ["correct", "incorrect"])
def test_build_feedback(rating):
    doc = schemas.build_feedback(schemas.new_id(), rating, comment="optional note")
    uuid.UUID(doc["feedback_id"])
    assert doc["rating"] == rating


@pytest.mark.parametrize("rating", ["good", "", None, 5])
def test_feedback_invalid_rating(rating):
    with pytest.raises(ValidationError):
        schemas.build_feedback(schemas.new_id(), rating)


def test_feedback_requires_valid_analysis_id():
    with pytest.raises(ValidationError):
        schemas.build_feedback("123", "correct")


# ---------------- model_versions ----------------

def test_build_model_version_minimal():
    doc = schemas.build_model_version("text", "example-model", "0.1.0")
    uuid.UUID(doc["model_id"])
    assert doc["metrics_metadata"] is None


@pytest.mark.parametrize("model_type", ["audio", "", None])
def test_model_version_invalid_type(model_type):
    with pytest.raises(ValidationError):
        schemas.build_model_version(model_type, "m", "1")


def test_model_version_requires_name_and_version():
    with pytest.raises(ValidationError):
        schemas.build_model_version("image", "", "1")
    with pytest.raises(ValidationError):
        schemas.build_model_version("image", "m", "")


def test_model_version_metrics_must_be_object():
    with pytest.raises(ValidationError):
        schemas.build_model_version("multimodal", "m", "1", metrics_metadata=[1, 2])
