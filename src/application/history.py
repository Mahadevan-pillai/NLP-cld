"""Consent-based analysis history and feedback (Task 16), through the Task 2 repositories only.

Policy (config/app.yaml `history`, decided by the developer in Task 16):
- nothing is stored unless the user gives consent for that analysis (consent is off by default);
- with consent, the submitted text / OCR text and the normalised text are stored, plus predictions,
  model versions, processing time, the moderation recommendation and OCR status;
- image binaries are NEVER stored; only width / height / format / mode / size_bytes;
- feedback (correct / incorrect + optional comment) is stored for saved analyses; it never retrains anything.

Database problems never break an analysis: every method returns {"ok": False, "message": <safe text>}
instead of raising, and the message never contains the connection string or server details.
"""

from src.application.logging_setup import log_event
from src.application.settings import load_app_config
from src.database import schemas

LABELS = ("misogynistic", "non_misogynistic")


class HistoryService:
    def __init__(self, db=None, app_config: dict | None = None):
        """db: a pymongo Database (or the test fake). None -> the shared Atlas connection, created lazily."""
        self._db = db
        self.cfg = (app_config or load_app_config())
        self._error = None

    # ------------------------------------------------------------------ connection
    def _database(self):
        if self._db is None:
            from src.database.connection import get_connection

            conn = get_connection()
            conn.ping()
            self._db = conn.db
        return self._db

    def _repos(self):
        from src.database.repositories import AnalysisHistoryRepository, UserFeedbackRepository

        db = self._database()
        analyses = AnalysisHistoryRepository(db, allowed_labels=LABELS)
        return analyses, UserFeedbackRepository(db, analysis_repository=analyses)

    @staticmethod
    def _safe(exc: Exception) -> str:
        from src.database.connection import DatabaseConfigError, DatabaseConnectionError

        if isinstance(exc, (DatabaseConfigError, DatabaseConnectionError)):
            return f"History is unavailable: {exc}"           # these messages are credential-free by design
        if isinstance(exc, (schemas.ValidationError, ValueError)):
            return f"Not saved: {exc}"
        return f"History is unavailable ({type(exc).__name__})."

    def status(self) -> dict:
        """{"ok": bool, "message": str} - a cheap connectivity check for the sidebar."""
        try:
            self._database()
            return {"ok": True, "message": "MongoDB connected"}
        except Exception as exc:
            return {"ok": False, "message": self._safe(exc)}

    # ------------------------------------------------------------------ saving
    @staticmethod
    def document(result: dict, source: str = "app") -> dict:
        """Repository arguments for an analysis result (no image, no explanation, no tensors)."""
        def pred(p):
            return {"label": p["name"], "score": float(p["score"])} if p and "name" in p else None

        models = result["models"]
        if result["analysis_type"] == "text":
            inp = {"original_text": result["input"]["original_text"], "normalized_text": result["input"]["normalized_text"]}
            preds = {"text_model": pred(models["text"]), "final": pred(result["prediction"])}
            ocr = None
        else:
            o = result["input"]["ocr"]
            inp = {"image_metadata": result["input"]["image_metadata"], "ocr_text": o["text"] or None,
                   "normalized_text": o["normalized_text"] or None}
            preds = {"text_model": pred(models["text"]), "image_model": pred(models["image"]),
                     "multimodal_model": pred(models["multimodal"]), "final": pred(result["prediction"])}
            ocr = {"status": o["status"], "route": o["route"], "confidence": o["confidence"], "engine": o["engine"],
                   "engine_version": o["engine_version"]}
        m = result["moderation"] or {}
        return {"analysis_type": result["analysis_type"], "input_data": {k: v for k, v in inp.items() if v is not None},
                "predictions": {k: v for k, v in preds.items() if v is not None}, "model_versions": result["model_versions"],
                "processing_time_ms": result["timings_ms"].get("total"),
                "extra": {"moderation": {"recommendation": m.get("recommendation"), "rule_id": m.get("rule_id"),
                                         "reason": m.get("reason")},
                          "ocr": ocr, "app_version": result["app_version"], "source": source}}

    def save(self, result: dict, consent: bool, source: str = "app") -> dict:
        """Store a successful analysis IF the user consented. Returns {"ok", "saved", "analysis_id", "message"}."""
        if not self.cfg["history"]["enabled"]:
            return {"ok": True, "saved": False, "analysis_id": None, "message": "History is disabled."}
        if not consent:
            return {"ok": True, "saved": False, "analysis_id": None, "message": "Not saved (no consent given)."}
        if result.get("status") == "error" or not result.get("prediction"):
            return {"ok": True, "saved": False, "analysis_id": None, "message": "Nothing to save (no prediction)."}
        try:
            analyses, _ = self._repos()
            doc = analyses.create_analysis(**self.document(result, source))
            log_event("history_saved", analysis_type=result["analysis_type"], saved=True)
            return {"ok": True, "saved": True, "analysis_id": doc["analysis_id"], "message": "Saved to history."}
        except Exception as exc:
            log_event("history_failed", analysis_type=result["analysis_type"], error_category=type(exc).__name__)
            return {"ok": False, "saved": False, "analysis_id": None, "message": self._safe(exc)}

    # ------------------------------------------------------------------ reading
    def recent(self, limit: int | None = None, analysis_type: str | None = None, final_label: str | None = None,
               since=None) -> dict:
        try:
            analyses, _ = self._repos()
            items = analyses.list_recent_analyses(limit or self.cfg["history"]["list_limit"], analysis_type,
                                                  final_label, since)
            return {"ok": True, "items": items, "message": ""}
        except Exception as exc:
            log_event("history_failed", stage="list", error_category=type(exc).__name__)
            return {"ok": False, "items": [], "message": self._safe(exc)}

    def get(self, analysis_id: str) -> dict:
        try:
            analyses, feedback = self._repos()
            doc = analyses.get_analysis_by_id(analysis_id)
            return {"ok": doc is not None, "item": doc, "feedback": feedback.get_feedback_for_analysis(analysis_id) if doc else [],
                    "message": "" if doc else "Analysis not found."}
        except Exception as exc:
            return {"ok": False, "item": None, "feedback": [], "message": self._safe(exc)}

    # ------------------------------------------------------------------ feedback
    def add_feedback(self, analysis_id: str, rating: str, comment: str | None = None) -> dict:
        if not self.cfg["feedback"]["enabled"]:
            return {"ok": False, "message": "Feedback is disabled."}
        comment = (comment or "").strip() or None
        if comment and len(comment) > self.cfg["feedback"]["comment_max_chars"]:
            return {"ok": False, "message": f"The comment is too long (max {self.cfg['feedback']['comment_max_chars']} characters)."}
        try:
            _, feedback = self._repos()
            doc = feedback.create_feedback(analysis_id, rating, comment)
            log_event("feedback_saved", saved=True)
            return {"ok": True, "feedback_id": doc["feedback_id"], "message": "Thank you, your feedback was saved."}
        except Exception as exc:
            log_event("feedback_failed", error_category=type(exc).__name__)
            return {"ok": False, "message": self._safe(exc)}
