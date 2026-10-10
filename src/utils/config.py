"""Load and sanity-check the project configuration file (config.yaml).

Usage:
    from src.utils.config import load_config
    config = load_config()
    seed = config["random_seed"]
"""

from pathlib import Path

import yaml

# Repository root = two levels above this file (src/utils/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

REQUIRED_SECTIONS = [
    "project",
    "paths",
    "dataset",
    "splitting",
    "random_seed",
    "text_preprocessing",
    "text_model",
    "image_model",
    "multimodal_model",
    "training",
    "evaluation",
    "database",
    "application",
]


def validate_config(config: dict) -> None:
    """Raise ValueError if the config is structurally invalid."""
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")

    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"Config is missing required sections: {missing}")

    seed = config["random_seed"]
    # bool is a subclass of int in Python, so exclude it explicitly.
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("random_seed must be an integer.")

    splitting = config["splitting"]
    ratios = [splitting.get(k) for k in ("train", "validation", "test")]
    if not all(isinstance(r, (int, float)) and 0 < r < 1 for r in ratios):
        raise ValueError("Split ratios must each be between 0 and 1.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}.")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read config.yaml, validate it, and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config
