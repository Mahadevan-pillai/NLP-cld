"""Document structure, construction and validation for the MongoDB collections.

Plain-Python validation (no extra dependency). Each collection has:
- a build_*() function that creates a new document with a UUID and a
  timezone-aware UTC timestamp, then validates it;
- a validate_*() function that raises ValidationError on bad documents.

Classification labels are NOT hard-coded: they have not been verified
yet (dataset inspection is a later task). Pass `allowed_labels` (e.g.
from config.yaml `dataset.labels`) to restrict them; with None, any
non-empty string is accepted.
"""

import math
import uuid
from datetime import datetime, timezone

ANALYSIS_TYPES = ("text", "meme")
MODEL_TYPES = ("text", "image", "multimodal")
FEEDBACK_RATINGS = ("correct", "incorrect")

# Which prediction / model slots make sense for each analysis type.
PREDICTION_KEYS = {
    "text": ("text_model", "final"),
    "meme": ("text_model", "image_model", "multimodal_model", "final"),
}
MODEL_SLOTS = ("text_model", "image_model", "multimodal_model")

# Which input fields each analysis type may contain.
INPUT_FIELDS = {
    "text": {"required": ("original_text",), "optional": ("normalized_text",)},
    "meme": {
        "required": (),
        "optional": ("image_reference", "image_metadata", "ocr_text", "normalized_text"),
    },
}

# Privacy: only basic, non-identifying image properties may be stored.
# This blocks accidentally saving EXIF data (which can include GPS location).
IMAGE_METADATA_KEYS = ("width", "height", "format", "mode", "size_bytes")

ANALYSIS_FIELDS = {
    "analysis_id", "analysis_type", "created_at", "processing_time_ms",
    "model_versions", "input", "predictions",
}
FEEDBACK_FIELDS = {"feedback_id", "analysis_id", "rating", "comment", "created_at"}
MODEL_VERSION_FIELDS = {
    "model_id", "model_type", "model_name", "architecture", "version",
    "training_dataset", "preprocessing_version", "created_at", "metrics_metadata",
}


class ValidationError(ValueError):
    """A database document does not have the expected structure."""


# ------------------------------------------------------------------
# IDs and timestamps
# ------------------------------------------------------------------

def new_id() -> str:
    """Random UUID4 as a string, used as a stable application-level ID."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    MongoDB stores dates with millisecond precision, so microseconds are
    truncated here to keep the in-memory value equal to the stored one.
    """
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=(now.microsecond // 1000) * 1000)


# ------------------------------------------------------------------
# Small reusable checks
# ------------------------------------------------------------------

def _check_uuid(value, name):
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a UUID string.")
    try:
        uuid.UUID(value)
    except ValueError:
        raise ValidationError(f"{name} must be a valid UUID string.") from None


def _check_timestamp(value, name="created_at"):
    if not isinstance(value, datetime):
        raise ValidationError(f"{name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{name} must be timezone-aware (UTC).")
    if value.utcoffset().total_seconds() != 0:
        raise ValidationError(f"{name} must be in UTC.")


def _check_non_empty_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string.")


def _check_optional_str(value, name):
    if value is not None and not isinstance(value, str):
        raise ValidationError(f"{name} must be a string or null.")


def _check_number(value, name, minimum=None):
    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise ValidationError(f"{name} must be >= {minimum}.")


def _check_known_fields(doc, allowed, name):
    unknown = set(doc) - set(allowed) - {"_id"}
    if unknown:
        raise ValidationError(f"{name} has unknown fields: {sorted(unknown)}")


def _check_no_binary(value, path="document"):
    """Reject raw bytes anywhere: image binaries must not go into documents."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ValidationError(f"{path} contains binary data; store a reference instead.")
    if isinstance(value, dict):
        for k, v in value.items():
            _check_no_binary(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_no_binary(v, f"{path}[{i}]")


def _check_prediction(pred, name, allowed_labels):
    if not isinstance(pred, dict):
        raise ValidationError(f"{name} must be an object with 'label' and optional 'score'.")
    _check_known_fields(pred, ("label", "score"), name)
    _check_non_empty_str(pred.get("label"), f"{name}.label")
    if allowed_labels is not None and pred["label"] not in allowed_labels:
        raise ValidationError(f"{name}.label is not one of the configured labels.")
    # "score" is any finite number: a probability for probabilistic models,
    # but e.g. an SVM decision value is not bounded to [0, 1].
    if pred.get("score") is not None:
        _check_number(pred["score"], f"{name}.score")


# ------------------------------------------------------------------
# analysis_history
# ------------------------------------------------------------------

def validate_analysis(doc: dict, allowed_labels=None) -> None:
    if not isinstance(doc, dict):
        raise ValidationError("analysis document must be a dict.")
    _check_known_fields(doc, ANALYSIS_FIELDS, "analysis")
    _check_no_binary(doc)

    _check_uuid(doc.get("analysis_id"), "analysis_id")
    analysis_type = doc.get("analysis_type")
    if analysis_type not in ANALYSIS_TYPES:
        raise ValidationError(f"analysis_type must be one of {ANALYSIS_TYPES}.")
    _check_timestamp(doc.get("created_at"))

    if doc.get("processing_time_ms") is not None:
        _check_number(doc["processing_time_ms"], "processing_time_ms", minimum=0)

    # input: required + optional fields depend on the analysis type
    inp = doc.get("input")
    if not isinstance(inp, dict):
        raise ValidationError("input must be an object.")
    spec = INPUT_FIELDS[analysis_type]
    _check_known_fields(inp, spec["required"] + spec["optional"], f"input ({analysis_type})")
    for key in spec["required"]:
        _check_non_empty_str(inp.get(key), f"input.{key}")
    for key in ("normalized_text", "ocr_text", "image_reference"):
        _check_optional_str(inp.get(key), f"input.{key}")
    ref = inp.get("image_reference")
    if isinstance(ref, str) and ref.lower().startswith("data:"):
        raise ValidationError("input.image_reference must be a reference, not an embedded image.")
    if inp.get("image_metadata") is not None:
        meta = inp["image_metadata"]
        if not isinstance(meta, dict):
            raise ValidationError("input.image_metadata must be an object.")
        _check_known_fields(meta, IMAGE_METADATA_KEYS, "input.image_metadata")

    # predictions: only slots that make sense for this analysis type
    preds = doc.get("predictions") or {}
    if not isinstance(preds, dict):
        raise ValidationError("predictions must be an object.")
    _check_known_fields(preds, PREDICTION_KEYS[analysis_type], f"predictions ({analysis_type})")
    for key, pred in preds.items():
        if pred is not None:
            _check_prediction(pred, f"predictions.{key}", allowed_labels)

    # model_versions: which model_id/version produced each prediction
    versions = doc.get("model_versions") or {}
    if not isinstance(versions, dict):
        raise ValidationError("model_versions must be an object.")
    _check_known_fields(versions, MODEL_SLOTS, "model_versions")
    for key, value in versions.items():
        _check_non_empty_str(value, f"model_versions.{key}")


def build_analysis(
    analysis_type: str,
    input_data: dict,
    predictions: dict | None = None,
    model_versions: dict | None = None,
    processing_time_ms: float | None = None,
    allowed_labels=None,
) -> dict:
    doc = {
        "analysis_id": new_id(),
        "analysis_type": analysis_type,
        "created_at": utc_now(),
        "processing_time_ms": processing_time_ms,
        "model_versions": model_versions or {},
        "input": input_data,
        "predictions": predictions or {},
    }
    validate_analysis(doc, allowed_labels)
    return doc


# ------------------------------------------------------------------
# user_feedback
# ------------------------------------------------------------------

def validate_feedback(doc: dict) -> None:
    if not isinstance(doc, dict):
        raise ValidationError("feedback document must be a dict.")
    _check_known_fields(doc, FEEDBACK_FIELDS, "feedback")
    _check_uuid(doc.get("feedback_id"), "feedback_id")
    _check_uuid(doc.get("analysis_id"), "analysis_id")
    if doc.get("rating") not in FEEDBACK_RATINGS:
        raise ValidationError(f"rating must be one of {FEEDBACK_RATINGS}.")
    _check_optional_str(doc.get("comment"), "comment")
    _check_timestamp(doc.get("created_at"))


def build_feedback(analysis_id: str, rating: str, comment: str | None = None) -> dict:
    doc = {
        "feedback_id": new_id(),
        "analysis_id": analysis_id,
        "rating": rating,
        "comment": comment,
        "created_at": utc_now(),
    }
    validate_feedback(doc)
    return doc


# ------------------------------------------------------------------
# model_versions
# ------------------------------------------------------------------

def validate_model_version(doc: dict) -> None:
    if not isinstance(doc, dict):
        raise ValidationError("model version document must be a dict.")
    _check_known_fields(doc, MODEL_VERSION_FIELDS, "model_version")
    _check_no_binary(doc)
    _check_uuid(doc.get("model_id"), "model_id")
    if doc.get("model_type") not in MODEL_TYPES:
        raise ValidationError(f"model_type must be one of {MODEL_TYPES}.")
    _check_non_empty_str(doc.get("model_name"), "model_name")
    _check_non_empty_str(doc.get("version"), "version")
    for key in ("architecture", "training_dataset", "preprocessing_version"):
        _check_optional_str(doc.get(key), key)
    if doc.get("metrics_metadata") is not None and not isinstance(doc["metrics_metadata"], dict):
        raise ValidationError("metrics_metadata must be an object or null.")
    _check_timestamp(doc.get("created_at"))


def build_model_version(
    model_type: str,
    model_name: str,
    version: str,
    architecture: str | None = None,
    training_dataset: str | None = None,
    preprocessing_version: str | None = None,
    metrics_metadata: dict | None = None,
) -> dict:
    doc = {
        "model_id": new_id(),
        "model_type": model_type,
        "model_name": model_name,
        "architecture": architecture,
        "version": version,
        "training_dataset": training_dataset,
        "preprocessing_version": preprocessing_version,
        "created_at": utc_now(),
        "metrics_metadata": metrics_metadata,
    }
    validate_model_version(doc)
    return doc
