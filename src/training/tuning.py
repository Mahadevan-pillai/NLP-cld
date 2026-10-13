"""Hyperparameter-search helpers (Task 12): search spaces, trial logging, best-trial selection.

Rules (decided before any search ran):
- Every trial is fitted on TRAIN and scored on VALIDATION. TEST is never used.
- Selection metric: validation macro-F1. Ties -> the EARLIER trial (deterministic).
- Only COMPLETED / EARLY_STOPPED trials can be selected. FAILED trials are logged, never dropped.
- Random search samples WITHOUT replacement from the full grid with a fixed seed.
"""

import csv
import itertools
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np

TRIAL_COLUMNS = ["trial_id", "search", "model", "modality", "parameters", "seed", "epochs", "best_epoch",
                 "validation_accuracy", "validation_macro_f1", "validation_weighted_f1", "train_macro_f1",
                 "training_time_s", "status", "note"]
SELECTABLE = ("COMPLETED", "EARLY_STOPPED")


def grid(space: dict) -> list[dict]:
    """All combinations of a {name: [values]} space, in a stable order."""
    names = list(space)
    return [dict(zip(names, values)) for values in itertools.product(*(space[n] for n in names))]


def grid_size(space: dict) -> int:
    return int(np.prod([len(v) for v in space.values()]))


def random_configs(space: dict, n: int, seed: int, exclude: list[dict] | None = None) -> list[dict]:
    """n distinct configurations sampled without replacement from the grid (seeded, reproducible).

    `exclude` removes configurations that are run separately (e.g. the baseline) before sampling.
    """
    full = [c for c in grid(space) if c not in (exclude or [])]
    if n >= len(full):
        return full
    return [full[i] for i in sorted(random.Random(seed).sample(range(len(full)), n))]


def params_json(params: dict) -> str:
    """Canonical JSON for a parameter dict (tuples -> lists, sorted keys)."""
    return json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}, sort_keys=True)


class TrialLog:
    """Collects trial rows and writes them as CSV. A failing trial is recorded as FAILED, not skipped."""

    def __init__(self):
        self.rows: list[dict] = []

    def run(self, trial_id: str, search: str, model: str, modality: str, params: dict, seed: int, fn) -> dict:
        """Call fn() -> dict of metrics (validation_macro_f1, ... , status optional). Logs and returns the row."""
        row = {c: None for c in TRIAL_COLUMNS}
        row.update(trial_id=trial_id, search=search, model=model, modality=modality,
                   parameters=params_json(params), seed=seed)
        start = time.perf_counter()
        try:
            result = fn() or {}
            row.update({k: v for k, v in result.items() if k in TRIAL_COLUMNS})
            row["status"] = result.get("status", "COMPLETED")
        except Exception as exc:                                      # recorded, never silently discarded
            row["status"] = "FAILED"
            row["note"] = f"{type(exc).__name__}: {exc}"[:300]
            traceback.print_exc()
        row["training_time_s"] = row["training_time_s"] if row["training_time_s"] is not None else \
            round(time.perf_counter() - start, 3)
        self.rows.append(row)
        return row

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)


def select_best(rows: list[dict], metric: str = "validation_macro_f1") -> dict:
    """Highest validation metric among selectable trials; ties -> earliest trial in the list."""
    candidates = [r for r in rows if r["status"] in SELECTABLE and r.get(metric) is not None]
    if not candidates:
        raise ValueError("no successful trial to select from")
    best = candidates[0]
    for r in candidates[1:]:
        if r[metric] > best[metric]:                                  # strictly better only -> earliest wins ties
            best = r
    return best


def seed_summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"n_seeds": int(len(v)), "mean": round(float(v.mean()), 4),
            "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
