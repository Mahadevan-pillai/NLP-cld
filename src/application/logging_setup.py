"""Safe application logging (Task 16).

Logged: operational events, model versions, durations, routes and safe error CATEGORIES.
Never logged: submitted text, OCR text, images, MongoDB URIs, tokens, or raw exception messages
(PyMongo / library messages can contain hosts or paths). `log_event` only accepts whitelisted keys.
"""

import logging
from logging.handlers import RotatingFileHandler

from src.utils.config import REPO_ROOT

LOGGER_NAME = "nlp_cld.app"
SAFE_KEYS = {"analysis_type", "model_versions", "duration_ms", "route", "recommendation", "rule_id",
             "error_category", "ocr_status", "saved", "component", "status", "count", "stage", "mode"}


def get_logger(file: str = "logs/app.log", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        path = REPO_ROOT / file
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def log_event(event: str, level: int = logging.INFO, **fields) -> dict:
    """Log `event` with whitelisted fields only; unknown keys are dropped (never raises)."""
    safe = {k: v for k, v in fields.items() if k in SAFE_KEYS}
    try:
        get_logger().log(level, "%s %s", event, " ".join(f"{k}={v}" for k, v in sorted(safe.items())))
    except Exception:          # logging must never break an analysis
        pass
    return safe
