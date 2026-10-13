"""Structured explanation object for the future UI (Task 15 design; Streamlit comes in Task 16).

    from src.explainability.explanation import explain_meme, explain_text_input
    explain_meme("meme.jpg")                 # OCR (Task 14) -> fallback route -> explanations -> moderation
    explain_text_input("some Hinglish text") # T05 (+ B07 second opinion) -> explanation -> moderation

Shape (every key always present; unavailable parts are None with a reason):

    {"mode", "prediction", "text_explanation", "image_explanation", "modality_comparison",
     "ocr", "moderation", "limitations", "disclaimer", "stored": False}

Nothing is written to disk and nothing is cached: user inputs are processed in memory only
(config/explainability.yaml cache.cache_user_uploads = false). Wording follows the responsible-language
rules: "model prediction", "potentially misogynistic", "flagged for review"; the model classifies
content, not people.
"""

from pathlib import Path

import yaml

from src.explainability.image import explain_image
from src.explainability.moderation import recommend
from src.explainability.multimodal import explain_multimodal
from src.explainability.text import explain_text
from src.utils.config import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "config" / "explainability.yaml"
GENERAL_LIMITATIONS = [
    "Explanations show model attribution and may not reflect human reasoning.",
    "Scores are uncalibrated and are not confidence percentages.",
    "The models were trained on one dataset (MIMIC); they can be wrong and can rely on shortcuts "
    "(for example, a pictured woman or certain words).",
    "Only misogyny is modelled: 'not flagged' does not mean the content is safe in other respects.",
]


def load_xai_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _prediction(p: dict, source: str) -> dict:
    return {"source": source, "model_version": p["model_version"], "predicted_label": p["predicted_label"],
            "predicted_name": p["predicted_name"], "score": p["score"], "score_type": p["score_type"],
            "calibrated": False}


def _ui_text(e: dict) -> dict:
    keep = ("model_version", "method", "tokens", "top_positive", "top_negative", "logit_difference",
            "completeness_error", "truncated", "processed_text_empty", "limitations")
    return {k: e.get(k) for k in keep}


def _ui_image(e: dict) -> dict:
    keep = ("model_version", "method", "heatmap", "heatmap_shape", "top_regions_towards_misogynistic",
            "top_regions_towards_non_misogynistic", "logit_difference", "limitations")
    return {k: e.get(k) for k in keep}


def explain_meme(image, caption: str | None = None, fast: bool = True, reader=None, config: dict | None = None) -> dict:
    """Full explanation for one meme. caption=None runs OCR (Task 14) and applies its fallback route."""
    cfg = config or load_xai_config()
    m = cfg["models"]
    window, stride = ((cfg["image"]["ui_window"], cfg["image"]["ui_stride"]) if fast
                      else (cfg["image"]["window"], cfg["image"]["stride"]))
    if caption is None:
        from src.ocr.pipeline import analyze_meme_input

        a = analyze_meme_input(image, reader=reader)
        ocr = {"status": a["ocr"]["status"], "confidence": a["ocr"]["confidence"], "route": a["decision"]["route"],
               "reason": a["decision"]["reason"], "engine": a["ocr"]["engine"], "engine_version": a["ocr"]["engine_version"]}
        text = a["ocr"]["text"]
    else:                       # same rule as Task 14: empty text never goes to the multimodal model
        empty = not caption.strip()
        ocr, text = {"status": "caption supplied", "confidence": None, "route": "image_only" if empty else "multimodal",
                     "reason": "empty caption" if empty else "caption supplied by the caller; OCR not run"}, caption
    out = {"mode": "meme", "prediction": None, "text_explanation": None, "image_explanation": None,
           "modality_comparison": None, "ocr": ocr, "moderation": None, "limitations": GENERAL_LIMITATIONS,
           "disclaimer": cfg["ui_disclaimer"], "stored": False}
    if ocr["route"] == "reject":
        out["moderation"] = recommend({"ocr_route": "reject"}, "meme", cfg["moderation"])
        return out
    img_e = explain_image(image, m["image"], window, stride)
    out["image_explanation"] = _ui_image(img_e)
    if ocr["route"] in ("image_only", "image_only_and_human_review"):
        out["prediction"] = _prediction(img_e, "image-only fallback (no reliable text)")
        out["moderation"] = recommend({"ocr_route": ocr["route"], "image_label": img_e["predicted_label"]}, "meme",
                                      cfg["moderation"])
        return out
    mm = explain_multimodal(image, text, m["multimodal"], m["text"], m["image"])
    out["text_explanation"] = _ui_text(explain_text(text, m["text"], cfg["text"]["steps"], cfg["text"]["top_k"]))
    out["prediction"] = {**mm["predictions"]["multimodal"], "source": "multimodal (primary prediction)"}
    out["modality_comparison"] = {"predictions": mm["predictions"], "agreement": mm["modality_agreement"],
                                  "disagreeing_models": mm["single_modality_models_disagreeing"],
                                  "ablation": mm["ablation"]}
    labels = {f"{k}_label": v["predicted_label"] for k, v in mm["predictions"].items()}
    out["moderation"] = recommend({"ocr_route": ocr["route"], **labels}, "meme", cfg["moderation"])
    return out


def explain_text_input(text: str, config: dict | None = None) -> dict:
    """Analyze Text mode: FINAL_TEXT_MODEL explanation, B07 second opinion, moderation recommendation."""
    cfg = config or load_xai_config()
    m = cfg["models"]
    t = explain_text(text, m["text"], cfg["text"]["steps"], cfg["text"]["top_k"])
    second = explain_text(text, m["traditional_text"], top_k=cfg["text"]["top_k"])
    return {"mode": "text", "prediction": _prediction(t, "text model (primary prediction)"),
            "text_explanation": _ui_text(t), "second_opinion": {**_prediction(second, "TF-IDF baseline"),
                                                                "top_positive": second["top_positive"]},
            "image_explanation": None, "modality_comparison": None, "ocr": None,
            "moderation": recommend({"text_label": t["predicted_label"], "second_text_label": second["predicted_label"],
                                     "processed_text_empty": t["processed_text_empty"]}, "text", cfg["moderation"]),
            "limitations": GENERAL_LIMITATIONS, "disclaimer": cfg["ui_disclaimer"], "stored": False}
