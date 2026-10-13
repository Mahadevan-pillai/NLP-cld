"""Application settings: config/app.yaml plus the frozen final-model selection (read, never changed)."""

from functools import lru_cache
from pathlib import Path

import yaml

from src.utils.config import REPO_ROOT, load_config

APP_CONFIG_PATH = REPO_ROOT / "config" / "app.yaml"


def load_app_config(path: Path = APP_CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def final_models() -> dict:
    """{"text": "T05", "image": "I03", "multimodal": "M04", "second_opinion_text": "B07"}.

    The finals come from config.yaml `final_models` (the Task 12/13 selection). The XAI config must
    name the same models; a mismatch is a configuration error, not something to resolve silently.
    """
    finals = load_config()["final_models"]
    xai = yaml.safe_load((REPO_ROOT / "config" / "explainability.yaml").read_text(encoding="utf-8"))["models"]
    models = {"text": finals["FINAL_TEXT_MODEL"], "image": finals["FINAL_IMAGE_MODEL"],
              "multimodal": finals["FINAL_MULTIMODAL_MODEL"]}
    if (xai["text"], xai["image"], xai["multimodal"]) != (models["text"], models["image"], models["multimodal"]):
        raise RuntimeError("config/explainability.yaml models differ from config.yaml final_models")
    models["second_opinion_text"] = load_app_config()["second_opinion_text_model"]
    return models


def supported_categories() -> dict:
    """Categories backed by trained models, derived from config/categories.yaml (never hard-coded).

    Returns {"enabled": [...], "disabled": [...], "unsupported": {name: reason}}.
    """
    cats = yaml.safe_load((REPO_ROOT / "config" / "categories.yaml").read_text(encoding="utf-8"))
    enabled, disabled = [], []
    for task, spec in cats["tasks"].items():
        names = [c["canonical"].title() for c in spec["categories"].values()]
        (enabled if spec.get("enabled") else disabled).append({"task": task, "type": spec["task_type"],
                                                               "categories": names})
    return {"enabled": enabled, "disabled": disabled, "unsupported": cats.get("unsupported", {}),
            "label_semantics": cats.get("label_semantics", "")}
