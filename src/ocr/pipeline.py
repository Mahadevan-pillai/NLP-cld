"""Real-world meme input pipeline (Task 14): image -> OCR -> Task 6 preprocessing -> routing decision.

    from src.ocr.pipeline import analyze_meme_input
    analyze_meme_input("meme.jpg")

It does NOT predict (that is the application task). It returns the OCR result, the OCR text in the
three Task 6 forms (raw / basic / hinglish_normalized, via the unchanged preprocess()) and the route the
application should take according to the fallback policy in config/ocr.yaml:

    multimodal                    OCR text usable -> image + text model
    image_only                    no text / OCR failure -> image-only model, "text could not be read" note
    image_only_and_human_review   very low OCR confidence -> image-only model + HUMAN REVIEW flag
    reject                        invalid / corrupt image -> no analysis

Empty OCR text is never silently sent to the multimodal model by this pipeline.
Nothing is stored: the image and its OCR text live only in memory for this call.
"""

from pathlib import Path

import yaml

from src.ocr.engine import extract_text_from_image
from src.preprocessing import PreprocessingConfig, preprocess
from src.utils.config import REPO_ROOT

OCR_CONFIG_PATH = REPO_ROOT / "config" / "ocr.yaml"
MODES = ("raw", "basic", "hinglish_normalized")


def load_ocr_config(path: Path = OCR_CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def preprocess_ocr_text(text: str, options: dict | None = None) -> dict:
    """OCR text in the three Task 6 forms, produced by the SAME preprocess() used for training."""
    cfg = PreprocessingConfig(**(options or {}))
    return {"ocr_text_raw": preprocess(text, "raw", cfg), "ocr_text_basic": preprocess(text, "basic", cfg),
            "ocr_text_hinglish_normalized": preprocess(text, "hinglish_normalized", cfg)}


def route(ocr: dict, processed: dict | None, policy: dict) -> dict:
    """Fallback decision from the OCR status, text and confidence (no model is run here)."""
    status = ocr["status"]
    if status == "invalid_image":
        return {"route": policy["invalid_image"], "reason": "invalid or corrupt image", "human_review": False}
    if status == "ocr_error":
        return {"route": policy["ocr_error"], "reason": "OCR failed", "human_review": False}
    if status == "no_text" or not (processed or {}).get("ocr_text_hinglish_normalized", "").strip():
        return {"route": policy["no_text"], "reason": "no text detected", "human_review": False}
    if ocr["confidence"] is not None and ocr["confidence"] < policy["min_confidence"]:
        return {"route": policy["low_confidence"], "reason": f"low OCR confidence ({ocr['confidence']:.2f})",
                "human_review": True}
    return {"route": policy["ok"], "reason": "OCR text available", "human_review": False}


def analyze_meme_input(image, preparation: str | None = None, preprocessing_options: dict | None = None,
                       reader=None, config: dict | None = None) -> dict:
    cfg = config or load_ocr_config()
    prep = preparation or cfg["preparation"] or "none"
    ocr = extract_text_from_image(image, preparation=prep, reader=reader)
    processed = preprocess_ocr_text(ocr["text"], preprocessing_options) if ocr["status"] in ("ok", "no_text") else None
    return {"ocr": ocr, "text": processed, "decision": route(ocr, processed, cfg["fallback"]), "stored": False}
