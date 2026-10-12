"""Category feasibility helpers and the proposed category configuration (Task 8.5).

The category config (config/categories.yaml) maps VERIFIED dataset labels to
canonical categories. Unsupported categories are listed but can never be
enabled through this loader.
"""

from pathlib import Path

import yaml

from src.utils.config import REPO_ROOT

CATEGORIES_PATH = REPO_ROOT / "config" / "categories.yaml"

# Training-feasibility tiers (documented in docs/MULTICATEGORY_FEASIBILITY_REPORT.md section 7).
# Judged on the canonical TRAIN and VALIDATION positives: train positives
# drive learning; validation positives determine how precisely the
# per-category score can be measured during development.
TIERS = (
    ("STRONG", 1000, 150),
    ("MODERATE", 500, 100),
    ("WEAK", 200, 50),
)


def feasibility_tier(train_positives: int, validation_positives: int) -> str:
    for name, min_train, min_val in TIERS:
        if train_positives >= min_train and validation_positives >= min_val:
            return name
    return "INSUFFICIENT"


def load_categories(path: Path = CATEGORIES_PATH, dataset_columns: list[str] | None = None) -> dict:
    """Load and validate the category config.

    Rules: enabled categories must come from VERIFIED dataset labels; their
    source columns must exist (if dataset_columns is given); unsupported
    categories cannot appear as enabled tasks.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    unsupported = set(config.get("unsupported", {}))
    for task_name, task in config["tasks"].items():
        if task["task_type"] not in ("binary", "multiclass", "multilabel"):
            raise ValueError(f"{task_name}: unknown task_type {task['task_type']!r}")
        for cat_name, cat in task["categories"].items():
            if cat["canonical"] in unsupported:
                raise ValueError(f"{task_name}.{cat_name}: {cat['canonical']} is listed as unsupported")
            if cat["status"] != "VERIFIED_DATASET_LABEL":
                raise ValueError(f"{task_name}.{cat_name}: only verified dataset labels may be configured")
            if dataset_columns is not None and cat["source_column"] not in dataset_columns:
                raise ValueError(f"{task_name}.{cat_name}: column {cat['source_column']!r} not in dataset")
    return config


def enabled_tasks(config: dict) -> list[str]:
    return [name for name, task in config["tasks"].items() if task["enabled"]]
