"""Moderation recommendation design (Task 15; for the future UI, not applied to anyone yet).

Outputs are limited to ALLOW / WARNING / HUMAN_REVIEW (project policy: no automatic BLOCK).
The logic is deterministic and uses only valid model OUTPUTS: predicted labels, agreement between
models and the Task 14 OCR route. It uses NO confidence thresholds, because the scores are not
calibrated.

The rules live in config/explainability.yaml (`moderation`): their ORDER, outcome ("then") and reason
text are configurable; each rule id's condition is defined once below. The first matching rule wins.
"""

from pathlib import Path

import yaml

from src.utils.config import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "config" / "explainability.yaml"
OUTPUTS = ("ALLOW", "WARNING", "HUMAN_REVIEW")
WORDING = {
    "ALLOW": "Not flagged by the models (model prediction).",
    "WARNING": "Potentially misogynistic content (model prediction). Shown with a warning; open to review.",
    "HUMAN_REVIEW": "Flagged for human review: the model outputs are uncertain or disagree.",
    None: "No recommendation: the input could not be analysed.",
}

# condition of each rule id; `x` is a dict of model outputs (labels 0/1, OCR route, empty-text flag)
MEME_CONDITIONS = {
    "M0": lambda x: x.get("ocr_route") == "reject",
    "M1": lambda x: x.get("ocr_route") == "image_only_and_human_review",
    "M2": lambda x: x.get("ocr_route") == "image_only" and x["image_label"] == 1,
    "M3": lambda x: x.get("ocr_route") == "image_only" and x["image_label"] == 0,
    "M4": lambda x: x["text_label"] == x["image_label"] == x["multimodal_label"] == 1,
    "M5": lambda x: x["text_label"] == x["image_label"] == x["multimodal_label"] == 0,
    "M6": lambda x: True,
}
TEXT_CONDITIONS = {
    "T0": lambda x: bool(x.get("processed_text_empty")),
    "T1": lambda x: x["text_label"] == x["second_text_label"] == 1,
    "T2": lambda x: x["text_label"] == x["second_text_label"] == 0,
    "T3": lambda x: True,
}


def load_policy(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["moderation"]


def validate_policy(policy: dict) -> None:
    for mode, conditions in (("meme_mode", MEME_CONDITIONS), ("text_mode", TEXT_CONDITIONS)):
        ids = [r["id"] for r in policy[mode]]
        unknown = set(ids) - set(conditions)
        if unknown or len(ids) != len(set(ids)):
            raise ValueError(f"{mode}: unknown or duplicate rule ids {sorted(unknown) or ids}")
        if ids[-1] not in ("M6", "T3"):
            raise ValueError(f"{mode}: the last rule must be the catch-all, got {ids[-1]}")
        for r in policy[mode]:
            if r["then"] is not None and r["then"] not in OUTPUTS:
                raise ValueError(f"{mode}/{r['id']}: outcome {r['then']!r} is not one of {OUTPUTS} (no BLOCK)")


def recommend(outputs: dict, mode: str = "meme", policy: dict | None = None) -> dict:
    """Recommendation for one input. meme mode needs text_label, image_label, multimodal_label (+ ocr_route);
    text mode needs text_label, second_text_label (+ processed_text_empty)."""
    policy = policy or load_policy()
    validate_policy(policy)
    rules, conditions = (policy["meme_mode"], MEME_CONDITIONS) if mode == "meme" else (policy["text_mode"], TEXT_CONDITIONS)
    for rule in rules:
        if conditions[rule["id"]](outputs):
            return {"recommendation": rule["then"], "rule_id": rule["id"], "reason": rule["reason"],
                    "wording": WORDING[rule["then"]], "automatic_block": False}
    raise RuntimeError("no rule matched (the catch-all rule is missing)")
