"""Unified application service (Task 16): analyze_text() / analyze_meme() for Streamlit and a future API.

Orchestrates EXISTING, validated components; no model logic is re-implemented here:
- validation          src.application.validation
- preprocessing       each model's own frozen Task 6 path (inside predict_text / predict_multimodal)
- inference           src.inference.text / image / multimodal (FINAL models from config.yaml; never substituted)
- OCR + fallback      src.ocr.pipeline.analyze_meme_input + route() (Task 14)
- explanations        src.explainability (Task 15): text IG automatically, image occlusion on demand
- moderation          src.explainability.moderation.recommend (ALLOW / WARNING / HUMAN_REVIEW; no BLOCK)

Every result is a plain, JSON-serialisable dict (schema below). Scores are UNCALIBRATED model scores.
Optional components (explanations, history) can fail without hiding the primary prediction.

Result schema (RESULT_SCHEMA_VERSION 1.0):
    analysis_type, status ("ok" | "limited" | "error"), error {category, message} | None, app_version,
    input {...}, prediction {...} | None, category {...}, models {text, image, multimodal, second_opinion},
    modality {...} | None, moderation {...} | None, explanation {...} | None, warnings [...],
    timings_ms {...}, model_versions {...}, disclaimer
"""

import time

import yaml

from src.application import artifacts, validation
from src.application.logging_setup import log_event
from src.application.settings import final_models, load_app_config, supported_categories
from src.explainability.moderation import load_policy, recommend
from src.inference import image as image_inference
from src.inference import text as text_inference
from src.inference.text import LABEL_NAMES
from src.utils.config import REPO_ROOT

RESULT_SCHEMA_VERSION = "1.0"


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


class AnalysisService:
    def __init__(self, app_config: dict | None = None, audit_rows: list[dict] | None = None, ocr_reader=None,
                 verify_checksums: bool | None = None):
        self.cfg = app_config or load_app_config()
        self.models = final_models()
        self.audit_rows = audit_rows if audit_rows is not None else artifacts.audit(verify_checksums)
        self.availability = artifacts.mode_availability(self.audit_rows)
        self.ocr_reader = ocr_reader                      # None -> the cached Task 14 EasyOCR reader
        self.policy = load_policy()
        self.xai = yaml.safe_load((REPO_ROOT / "config" / "explainability.yaml").read_text(encoding="utf-8"))
        cats = supported_categories()
        self.task_name = cats["enabled"][0]["categories"][0] if cats["enabled"] else None
        self.subcategory_note = ("not available: no sub-category model is trained ("
                                 + ", ".join(c for d in cats["disabled"] for c in d["categories"]) + ")") if cats["disabled"] else None

    # ------------------------------------------------------------------ helpers
    def _display(self, label: int) -> str:
        return self.cfg["labels"][LABEL_NAMES[label]]

    def _pred(self, p: dict, role: str, used_input: str) -> dict:
        return {"model_version": p["model_version"], "role": role, "used_input": used_input,
                "label": p["predicted_label"], "name": p["predicted_name"], "display": self._display(p["predicted_label"]),
                "score": p["score"], "score_type": p["score_type"], "calibrated": False}

    def _category(self, label: int | None) -> dict:
        if label is None:
            return {"task": "misogyny detection (binary)", "label": None, "subcategories": self.subcategory_note}
        return {"task": "misogyny detection (binary)",
                "label": f"{self.task_name} (potentially misogynistic)" if label == 1 else
                "no misogyny detected by the model (other harms are not modelled)",
                "subcategories": self.subcategory_note}

    def _normalized(self, text: str) -> tuple[str, str | None]:
        """The text exactly as the FINAL text model sees it (its own Task 6 mode) + a truncation warning."""
        from src.models.transformer_text import preprocess_all

        kind, entry, obj = text_inference._load(self.models["text"], str(text_inference.REGISTRY_PATH))
        processed = preprocess_all([text], entry["preprocessing_mode"], entry.get("preprocessing_options"))[0]
        warning = validation.token_warning(obj[1], processed, entry["max_length"]) if kind == "fine_tuned" else None
        return processed, warning

    def _base(self, analysis_type: str) -> dict:
        return {"schema_version": RESULT_SCHEMA_VERSION, "analysis_type": analysis_type, "status": "ok", "error": None,
                "app_version": self.cfg["app_version"], "input": {}, "prediction": None, "category": self._category(None),
                "models": {"text": None, "image": None, "multimodal": None, "second_opinion": None}, "modality": None,
                "moderation": None, "explanation": None, "warnings": [], "timings_ms": {}, "model_versions": {},
                "disclaimer": self.cfg["xai"]["disclaimer"]}

    def _fail(self, result: dict, category: str, message: str, t0: float) -> dict:
        result.update(status="error", error={"category": category, "message": message})
        result["timings_ms"]["total"] = _ms(t0)
        log_event("analysis_failed", analysis_type=result["analysis_type"], error_category=category,
                  duration_ms=result["timings_ms"]["total"])
        return result

    # ------------------------------------------------------------------ start-up
    def warm_up(self) -> dict:
        """Load every available model once (cached by the inference modules); returns seconds per component."""
        from PIL import Image

        out = {}
        if self.availability["text"][0]:
            t0 = time.perf_counter()
            text_inference.predict_text("warm up", self.models["text"])
            text_inference.predict_text("warm up", self.models["second_opinion_text"])
            out["text models (T05 + B07)"] = round(time.perf_counter() - t0, 2)
        if self.availability["meme"][0]:
            from src.inference.multimodal import predict_multimodal

            img = Image.new("RGB", (64, 64), (128, 128, 128))
            t0 = time.perf_counter()
            image_inference.predict_image(img, self.models["image"])
            out["image model (I03 + SigLIP)"] = round(time.perf_counter() - t0, 2)
            t0 = time.perf_counter()
            predict_multimodal(img, "warm up", self.models["multimodal"])
            out["multimodal model (M04)"] = round(time.perf_counter() - t0, 2)
            if self.ocr_reader is None:
                from src.ocr.engine import get_reader

                t0 = time.perf_counter()
                get_reader()
                out["OCR reader (EasyOCR)"] = round(time.perf_counter() - t0, 2)
        log_event("warm_up", status="done", count=len(out))
        return out

    # ------------------------------------------------------------------ text
    def analyze_text(self, text, explain: bool = True) -> dict:
        t0 = time.perf_counter()
        result = self._base("text")
        try:
            artifacts.require("text", self.audit_rows)
        except artifacts.ArtifactUnavailableError as exc:
            return self._fail(result, "artifact_unavailable", str(exc), t0)
        try:
            cleaned, warnings = validation.validate_text(text, self.cfg["text_input"]["max_chars"],
                                                         self.cfg["text_input"]["min_letters"])
        except validation.InputError as exc:
            return self._fail(result, "invalid_input", str(exc), t0)
        result["warnings"] += warnings
        try:
            t1 = time.perf_counter()
            primary = text_inference.predict_text(cleaned, self.models["text"])
            second = text_inference.predict_text(cleaned, self.models["second_opinion_text"])
            normalized, trunc = self._normalized(cleaned)
            result["timings_ms"]["inference"] = _ms(t1)
        except Exception as exc:                          # safe category only; no raw message to the user
            return self._fail(result, "inference_error", f"The text model failed ({type(exc).__name__}). "
                                                         "No prediction is shown.", t0)
        if trunc:
            result["warnings"].append(trunc)
        result["input"] = {"original_text": cleaned, "normalized_text": normalized,
                           "preprocessing": f"Task 6 v{primary['preprocessing_version']} ({primary['preprocessing_mode']})"}
        result["models"]["text"] = self._pred(primary, "FINAL_TEXT_MODEL (primary)", "text")
        result["models"]["second_opinion"] = self._pred(second, "TF-IDF baseline (second opinion for moderation)", "text")
        result["prediction"] = {**result["models"]["text"], "source": "text model"}
        result["category"] = self._category(primary["predicted_label"])
        result["moderation"] = recommend({"text_label": primary["predicted_label"],
                                          "second_text_label": second["predicted_label"],
                                          "processed_text_empty": primary["processed_text_empty"]}, "text", self.policy)
        result["model_versions"] = {"text_model": primary["model_version"]}
        if explain:
            result["explanation"] = {"text": self.explain_text(cleaned)}
        result["timings_ms"]["total"] = _ms(t0)
        log_event("analysis", analysis_type="text", model_versions=self.models["text"],
                  recommendation=result["moderation"]["recommendation"], rule_id=result["moderation"]["rule_id"],
                  duration_ms=result["timings_ms"]["total"])
        return result

    def explain_text(self, text: str) -> dict:
        """T05 word attributions (Integrated Gradients); failures return an error entry, never raise."""
        from src.explainability.text import explain_text

        t0 = time.perf_counter()
        try:
            e = explain_text(text, self.models["text"], self.xai["text"]["steps"], self.xai["text"]["top_k"])
            return {"status": "ok", "method": e["method"], "tokens": e["tokens"], "top_positive": e["top_positive"],
                    "top_negative": e["top_negative"], "completeness_error": e["completeness_error"],
                    "truncated": e["truncated"], "limitations": e["limitations"], "duration_ms": _ms(t0)}
        except Exception as exc:
            log_event("explanation_failed", component="text", error_category=type(exc).__name__)
            return {"status": "error", "message": "The text explanation could not be computed.", "duration_ms": _ms(t0)}

    # ------------------------------------------------------------------ meme
    def analyze_meme(self, data: bytes) -> dict:
        from src.explainability.multimodal import explain_multimodal
        from src.ocr.pipeline import analyze_meme_input, load_ocr_config, route

        t0 = time.perf_counter()
        result = self._base("meme")
        try:
            artifacts.require("meme", self.audit_rows)
        except artifacts.ArtifactUnavailableError as exc:
            return self._fail(result, "artifact_unavailable", str(exc), t0)
        up = self.cfg["upload"]
        try:
            img, meta = validation.validate_upload(data, tuple(up["allowed_formats"]), up["max_upload_mb"], up["max_pixels"])
        except validation.InputError as exc:
            return self._fail(result, "invalid_input", str(exc), t0)
        result["input"]["image_metadata"] = meta
        # ---- OCR (Task 14) with its pre-declared fallback route
        t1 = time.perf_counter()
        try:
            a = analyze_meme_input(img, reader=self.ocr_reader)
            ocr, decision = a["ocr"], a["decision"]
        except Exception as exc:
            ocr = {"status": "ocr_error", "text": "", "confidence": None, "engine": "easyocr", "engine_version": None,
                   "detected_regions": [], "error": type(exc).__name__}
            decision = route(ocr, None, load_ocr_config()["fallback"])
        result["timings_ms"]["ocr"] = _ms(t1)
        ocr_text = ocr.get("text") or ""
        normalized = self._normalized(ocr_text)[0] if ocr_text.strip() else ""
        result["input"]["ocr"] = {"status": ocr["status"], "text": ocr_text, "normalized_text": normalized,
                                  "confidence": ocr.get("confidence"), "regions": len(ocr.get("detected_regions") or []),
                                  "engine": ocr.get("engine"), "engine_version": ocr.get("engine_version"),
                                  "route": decision["route"], "reason": decision["reason"]}
        if decision["route"] == "reject":
            return self._fail(result, "invalid_input", "The image could not be read; please upload another file.", t0)
        # ---- models
        t2 = time.perf_counter()
        try:
            if decision["route"] == "multimodal":
                mm = explain_multimodal(img, ocr_text, self.models["multimodal"], self.models["text"], self.models["image"])
                preds = mm["predictions"]
                result["models"]["text"] = self._pred(preds["text"], "FINAL_TEXT_MODEL", "OCR text only")
                result["models"]["image"] = self._pred(preds["image"], "FINAL_IMAGE_MODEL", "image only (no text)")
                result["models"]["multimodal"] = self._pred(preds["multimodal"], "FINAL_MULTIMODAL_MODEL (primary)",
                                                            "image + OCR text")
                result["modality"] = {"agreement": mm["modality_agreement"],
                                      "disagreeing": mm["single_modality_models_disagreeing"],
                                      "dependence": mm["ablation"]["dependence"],
                                      "label_changes_without_text": mm["ablation"]["label_changes_without_text"],
                                      "label_changes_without_image": mm["ablation"]["label_changes_without_image"]}
                result["prediction"] = {**result["models"]["multimodal"], "source": "multimodal model (primary)"}
                labels = {f"{k}_label": result["models"][k]["label"] for k in ("text", "image", "multimodal")}
                result["model_versions"] = {"text_model": self.models["text"], "image_model": self.models["image"],
                                            "multimodal_model": self.models["multimodal"]}
            else:
                p = image_inference.predict_image(img, self.models["image"])
                result["models"]["image"] = self._pred(p, "FINAL_IMAGE_MODEL (fallback primary)", "image only (no text)")
                reason = decision["reason"]
                result["models"]["text"] = {"unavailable": f"not run: {reason}"}
                result["models"]["multimodal"] = {"unavailable": f"not run: {reason} (empty or unreliable text is never sent "
                                                                 "to the multimodal model)"}
                result["prediction"] = {**result["models"]["image"], "source": "image-only fallback"}
                result["status"] = "limited"
                result["warnings"].append(f"Limited analysis: {reason}. Only the image model was used; the caption "
                                          "could not be analysed.")
                labels = {"image_label": p["predicted_label"]}
                result["model_versions"] = {"image_model": self.models["image"]}
        except Exception as exc:
            return self._fail(result, "inference_error", f"A model failed ({type(exc).__name__}). No prediction is shown.", t0)
        result["timings_ms"]["inference"] = _ms(t2)
        result["category"] = self._category(result["prediction"]["label"])
        result["moderation"] = recommend({"ocr_route": decision["route"], **labels}, "meme", self.policy)
        result["timings_ms"]["total"] = _ms(t0)
        log_event("analysis", analysis_type="meme", route=decision["route"], ocr_status=ocr["status"],
                  model_versions="/".join(result["model_versions"].values()),
                  recommendation=result["moderation"]["recommendation"], rule_id=result["moderation"]["rule_id"],
                  duration_ms=result["timings_ms"]["total"])
        return result

    def explain_meme(self, data: bytes, ocr_text: str | None) -> dict:
        """On-demand explanation for a meme: I03 occlusion map (+ overlay) and T05 word attributions of the OCR text."""
        from src.explainability.image import explain_image, overlay

        out = {"image": None, "text": None}
        t0 = time.perf_counter()
        try:
            img, _ = validation.validate_upload(data, tuple(self.cfg["upload"]["allowed_formats"]),
                                                self.cfg["upload"]["max_upload_mb"], self.cfg["upload"]["max_pixels"])
            e = explain_image(img, self.models["image"], self.cfg["xai"]["image_window"], self.cfg["xai"]["image_stride"])
            import numpy as np

            out["image"] = {"status": "ok", "method": e["method"], "heatmap": e["heatmap"],
                            "top_regions_towards_misogynistic": e["top_regions_towards_misogynistic"],
                            "top_regions_towards_non_misogynistic": e["top_regions_towards_non_misogynistic"],
                            "overlay": overlay(img, np.array(e["heatmap"])), "limitations": e["limitations"],
                            "duration_ms": _ms(t0)}
        except Exception as exc:
            log_event("explanation_failed", component="image", error_category=type(exc).__name__)
            out["image"] = {"status": "error", "message": "The image explanation could not be computed.", "duration_ms": _ms(t0)}
        if ocr_text and ocr_text.strip():
            out["text"] = self.explain_text(ocr_text)
        return out
