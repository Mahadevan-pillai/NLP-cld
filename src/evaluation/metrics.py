"""Classification metrics with explicit averaging conventions.

Conventions (binary task, positive class = 1 = misogynistic):
  precision / recall / f1     -> for the POSITIVE class only (average="binary")
  macro_*                     -> unweighted mean over both classes (PRIMARY: macro_f1)
  weighted_f1                 -> mean over classes weighted by true support
  per-class metrics           -> reported separately for each class
Confusion matrix: rows = true label, columns = predicted label, order [0, 1].
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

LABELS = [0, 1]


def classification_metrics(y_true, y_pred, names: dict[int, str]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    for i, label in enumerate(LABELS):
        name = names[label]
        out[f"precision_{name}"], out[f"recall_{name}"] = p[i], r[i]
        out[f"f1_{name}"], out[f"support_{name}"] = f[i], int(s[i])
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    out.update({"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def bootstrap_macro_f1(y_true, y_pred, n_resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% percentile bootstrap interval of macro-F1 on one evaluation set.

    Describes sampling uncertainty of the score only. It is NOT a
    significance test between models.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
              for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)
