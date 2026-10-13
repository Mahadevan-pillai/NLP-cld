"""Repository and index tests using the in-memory FakeDatabase (no MongoDB needed)."""

import time

import pytest

from src.database.indexes import INDEXES, ensure_indexes
from src.database.repositories import (
    AnalysisHistoryRepository,
    ModelVersionRepository,
    UserFeedbackRepository,
)
from src.database.schemas import ValidationError, new_id


@pytest.fixture
def analyses(fake_db):
    return AnalysisHistoryRepository(fake_db)


def make_text(repo, text="example"):
    return repo.create_analysis("text", {"original_text": text})


# ---------------- AnalysisHistoryRepository ----------------

def test_create_and_get_analysis_roundtrip(analyses, fake_db):
    created = make_text(analyses)
    assert "_id" not in created
    fetched = analyses.get_analysis_by_id(created["analysis_id"])
    assert fetched == created
    assert "_id" in fake_db["analysis_history"].docs[0]  # MongoDB _id kept internally


def test_get_missing_analysis_returns_none(analyses):
    assert analyses.get_analysis_by_id(new_id()) is None


def test_invalid_analysis_is_not_inserted(analyses, fake_db):
    with pytest.raises(ValidationError):
        analyses.create_analysis("text", {})
    assert fake_db["analysis_history"].docs == []


def test_list_recent_newest_first_with_limit_and_filter(analyses):
    first = make_text(analyses, "one")
    time.sleep(0.002)  # timestamps have millisecond precision
    meme = analyses.create_analysis("meme", {"ocr_text": "two"})
    time.sleep(0.002)
    last = make_text(analyses, "three")

    recent = analyses.list_recent_analyses()
    assert [d["analysis_id"] for d in recent] == [last["analysis_id"], meme["analysis_id"], first["analysis_id"]]
    assert len(analyses.list_recent_analyses(limit=2)) == 2
    assert [d["analysis_id"] for d in analyses.list_recent_analyses(analysis_type="meme")] == [meme["analysis_id"]]


@pytest.mark.parametrize("limit", [0, -1, 101, 2.5, True])
def test_list_recent_rejects_bad_limit(analyses, limit):
    with pytest.raises(ValueError):
        analyses.list_recent_analyses(limit=limit)


def test_list_recent_rejects_bad_type(analyses):
    with pytest.raises(ValueError):
        analyses.list_recent_analyses(analysis_type="video")


def test_allowed_labels_passed_through(fake_db):
    repo = AnalysisHistoryRepository(fake_db, allowed_labels=["a"])
    with pytest.raises(ValidationError):
        repo.create_analysis("text", {"original_text": "x"}, predictions={"final": {"label": "b"}})


# ---------------- UserFeedbackRepository ----------------

def test_feedback_create_and_get(fake_db, analyses):
    analysis = make_text(analyses)
    feedback = UserFeedbackRepository(fake_db, analysis_repository=analyses)
    fb = feedback.create_feedback(analysis["analysis_id"], "incorrect", comment="wrong label")
    assert feedback.get_feedback_for_analysis(analysis["analysis_id"]) == [fb]
    assert feedback.get_feedback_for_analysis(new_id()) == []


def test_feedback_for_unknown_analysis_rejected(fake_db):
    feedback = UserFeedbackRepository(fake_db)
    with pytest.raises(ValueError, match="does not exist"):
        feedback.create_feedback(new_id(), "correct")
    assert fake_db["user_feedback"].docs == []


# ---------------- ModelVersionRepository ----------------

def test_model_version_register_get_list(fake_db):
    repo = ModelVersionRepository(fake_db)
    text_model = repo.register_model_version("text", "example-text", "0.1.0", architecture="tfidf+logreg")
    repo.register_model_version("image", "example-image", "0.1.0")

    assert repo.get_model_version(text_model["model_id"]) == text_model
    assert repo.get_model_version(new_id()) is None
    assert len(repo.list_model_versions()) == 2
    assert [m["model_id"] for m in repo.list_model_versions(model_type="text")] == [text_model["model_id"]]
    with pytest.raises(ValueError):
        repo.list_model_versions(model_type="audio")


def test_custom_collection_name(fake_db):
    repo = ModelVersionRepository(fake_db, collection_name="custom_models")
    repo.register_model_version("text", "m", "1")
    assert len(fake_db["custom_models"].docs) == 1


# ---------------- ensure_indexes ----------------

def test_ensure_indexes_creates_expected_and_is_idempotent(fake_db):
    first = ensure_indexes(fake_db)
    second = ensure_indexes(fake_db)
    assert first == second
    assert set(first) == {"analysis_history", "user_feedback", "model_versions"}

    for logical, specs in INDEXES.items():
        assert len(fake_db[logical].indexes) == len(specs)
    unique = {name for name, spec in fake_db["analysis_history"].indexes.items() if spec["unique"]}
    assert unique == {"analysis_id_unique"}
    assert fake_db["user_feedback"].indexes["feedback_id_unique"]["unique"]
    assert fake_db["model_versions"].indexes["model_id_unique"]["unique"]
