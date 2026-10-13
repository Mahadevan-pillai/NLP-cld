"""Final-evaluation helpers (Task 13): checksums, paired bootstrap, McNemar test, consistency checks.

Statistical plan (fixed in results/final/model_freeze_manifest.json BEFORE the test set was read):
- 95% percentile bootstrap intervals, 10,000 resamples, seed 42, resampling TEST samples with
  replacement; for model DIFFERENCES the same resampled indices are used for both models (paired).
- McNemar's exact test (two-sided binomial test on the discordant pairs) for paired error rates.
- A difference is called "statistically significant" only for the pre-specified primary comparisons,
  only if the Holm-adjusted McNemar p < 0.05 AND the paired bootstrap CI of the macro-F1 difference
  excludes 0. Everything else is reported descriptively.
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def sha256_path(path: Path) -> str:
    """SHA-256 of a file, or of a folder (sorted relative paths + file hashes)."""
    path = Path(path)
    if path.is_file():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        h = hashlib.sha256()
        for p in sorted(q for q in path.rglob("*") if q.is_file()):
            h.update(p.relative_to(path).as_posix().encode())
            h.update(sha256_path(p).encode())
        return h.hexdigest()
    raise FileNotFoundError(path)


def macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary macro-F1 from counts (identical to sklearn f1_score(average='macro', zero_division=0))."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return float((f1_pos + f1_neg) / 2)


def _macro_f1_batched(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Macro-F1 for every bootstrap resample at once (idx: resamples x n)."""
    yt, yp = y[idx], p[idx]
    tp = ((yt == 1) & (yp == 1)).sum(1)
    tn = ((yt == 0) & (yp == 0)).sum(1)
    fp = ((yt == 0) & (yp == 1)).sum(1)
    fn = ((yt == 1) & (yp == 0)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f_pos + f_neg) / 2


def bootstrap_indices(n: int, resamples: int = 10_000, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def bootstrap_ci(y_true, y_pred, idx: np.ndarray, level: float = 0.95) -> dict:
    y, p = np.asarray(y_true), np.asarray(y_pred)
    values = _macro_f1_batched(y, p, idx)
    lo, hi = np.percentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"macro_f1": round(macro_f1_fast(y, p), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}


def paired_bootstrap_difference(y_true, pred_a, pred_b, idx: np.ndarray, level: float = 0.95) -> dict:
    """Macro-F1(a) - macro-F1(b) on the same resampled test samples."""
    y, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    diffs = _macro_f1_batched(y, a, idx) - _macro_f1_batched(y, b, idx)
    lo, hi = np.percentile(diffs, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"difference": round(macro_f1_fast(y, a) - macro_f1_fast(y, b), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "share_of_resamples_a_better": round(float((diffs > 0).mean()), 4)}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test: b = a right & b wrong, c = a wrong & b right; two-sided binomial p on b of b + c."""
    y, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((a != y) & (b == y)).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5, alternative="two-sided").pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "discordant": n, "p_value": p}


def holm(p_values: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a dict {name: p}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def confusion_consistent(metrics: dict, n: int) -> bool:
    """Confusion-matrix cells must sum to the evaluated sample count, and supports must match rows."""
    total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
    return (total == n == metrics["n"] and metrics["support_non_misogynistic"] == metrics["tn"] + metrics["fp"]
            and metrics["support_misogynistic"] == metrics["fn"] + metrics["tp"])
