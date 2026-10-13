"""Train and evaluate the TF-IDF baselines on TRAIN / VALIDATION only.

The TEST split is never loaded here (see load_development_data). Final test
evaluation belongs to a later, explicitly authorised step.
"""

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.splits import load_split, load_split_assignments
from src.evaluation.metrics import bootstrap_macro_f1, classification_metrics
from src.models.text_baselines import build_pipeline
from src.preprocessing import hinglish

INPUT_COLUMN = "text_raw"   # pipelines preprocess internally from raw text
STORED_COLUMN = {"raw": "text_raw", "basic": "text_basic", "hinglish_normalized": "text_hinglish_normalized"}
WATERMARK_RE = re.compile(r"doublemean", re.IGNORECASE)


def load_development_data(config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, validation) from the canonical split. The test split is not touched."""
    train, val = load_split("train", config=config), load_split("validation", config=config)
    assignments = load_split_assignments(config)
    expected = assignments["split"].value_counts()
    if len(train) != expected["train"] or len(val) != expected["validation"]:
        raise RuntimeError("train/validation sizes differ from the canonical split")
    if set(train["sample_id"]) & set(val["sample_id"]):
        raise RuntimeError("train and validation overlap")
    for part in (train, val):
        if not part["target_label"].isin(["0", "1"]).all():
            raise RuntimeError("invalid labels")
    return train, val


@dataclass
class ExperimentResult:
    experiment_id: str
    classifier: str
    mode: str
    pipeline: object
    metrics: dict
    subsets: dict
    macro_f1_ci: tuple
    train_seconds: float
    predict_seconds: float
    n_features: int
    val_pred: np.ndarray
    val_scores: np.ndarray
    converged: bool


def subset_masks(val: pd.DataFrame) -> dict:
    has_text = val["text_raw"].str.strip() != ""
    watermark = val["text_raw"].str.contains(WATERMARK_RE)
    return {"all": np.ones(len(val), bool), "with_text": has_text.to_numpy(),
            "without_watermark": (~watermark).to_numpy()}


def run_experiment(experiment_id: str, classifier: str, mode: str, train: pd.DataFrame, val: pd.DataFrame,
                   config: dict, names: dict[int, str]) -> ExperimentResult:
    seed = config["random_seed"]
    pipeline = build_pipeline(classifier, mode, config, seed)
    y_train = train["target_label"].astype(int).to_numpy()
    y_val = val["target_label"].astype(int).to_numpy()

    start = time.perf_counter()
    pipeline.fit(train[INPUT_COLUMN].tolist(), y_train)          # TF-IDF + classifier fitted on TRAIN only
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pred = pipeline.predict(val[INPUT_COLUMN].tolist())          # validation is only transformed
    predict_seconds = time.perf_counter() - start
    scores = pipeline.decision_function(val[INPUT_COLUMN].tolist())

    clf = pipeline.named_steps["clf"]
    converged = bool(np.all(np.asarray(clf.n_iter_) < clf.max_iter))
    subsets = {}
    for name, mask in subset_masks(val).items():
        m = classification_metrics(y_val[mask], pred[mask], names)
        subsets[name] = {"n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    return ExperimentResult(
        experiment_id, classifier, mode, pipeline, classification_metrics(y_val, pred, names), subsets,
        bootstrap_macro_f1(y_val, pred, config["text_model"]["baselines"]["bootstrap_resamples"], seed),
        train_seconds, predict_seconds, len(pipeline.named_steps["tfidf"].vocabulary_), pred, scores, converged)


def check_preprocessing_matches_task6(pipeline, data: pd.DataFrame, mode: str) -> bool:
    """The pipeline's own preprocessing must reproduce the stored Task 6 column exactly."""
    produced = pipeline.named_steps["preprocess"].transform(data[INPUT_COLUMN].tolist())
    return produced == data[STORED_COLUMN[mode]].tolist()


def top_features(result: ExperimentResult, names: dict[int, str], top: int = 25) -> pd.DataFrame:
    """Largest positive / negative linear weights (model-behaviour diagnostics, not causal)."""
    vocab = result.pipeline.named_steps["tfidf"].get_feature_names_out()
    coef = result.pipeline.named_steps["clf"].coef_.ravel()
    order = np.argsort(coef)
    rows = []
    for rank, i in enumerate(order[::-1][:top], start=1):
        rows.append({"experiment_id": result.experiment_id, "direction": names[1], "rank": rank,
                     "feature": vocab[i], "weight": round(float(coef[i]), 4)})
    for rank, i in enumerate(order[:top], start=1):
        rows.append({"experiment_id": result.experiment_id, "direction": names[0], "rank": rank,
                     "feature": vocab[i], "weight": round(float(coef[i]), 4)})
    table = pd.DataFrame(rows)
    table["artifact_flag"] = table["feature"].map(feature_flag)
    return table


def feature_flag(feature: str) -> str:
    """Flag features that look like dataset artefacts rather than content."""
    if "doublemean" in feature:
        return "watermark_handle"
    if re.search(r"\buser\b", feature):
        return "mention_placeholder"
    if re.fullmatch(r"[\d ]+", feature):
        return "number_only"
    if any(len(t) == 1 for t in feature.split()):
        return "single_character_token"
    return ""


def error_table(val: pd.DataFrame, results: list[ExperimentResult], best_id: str) -> pd.DataFrame:
    """Per-sample validation diagnostics for every sample the best model got wrong.

    Contains IDs, labels, predictions and computed flags only (no text).
    """
    y = val["target_label"].astype(int).to_numpy()
    preds = {r.experiment_id: r.val_pred for r in results}
    best = preds[best_id]
    wrong_count = sum((p != y).astype(int) for p in preds.values())
    raw = val["text_raw"]
    rule_hits = [sum(1 for _ in hinglish.load_rules().pattern.finditer(t)) for t in val["text_basic"]]
    table = pd.DataFrame({
        "sample_id": val["sample_id"], "true_label": y, "pred_best": best,
        "error_type": np.where((best == 1) & (y == 0), "false_positive",
                               np.where((best == 0) & (y == 1), "false_negative", "correct")),
        "models_wrong_of_6": wrong_count,
        **{f"pred_{k}": v for k, v in preds.items()},
        "has_text": raw.str.strip() != "",
        "words": raw.str.split().str.len(),
        "very_short_le3_words": raw.str.split().str.len().between(1, 3),
        "has_negation": [hinglish.count_negations(t) > 0 for t in raw],
        "hinglish_rule_hits": rule_hits,
        "has_devanagari": raw.str.contains(r"[ऀ-ॿ]"),
        "has_watermark": raw.str.contains(WATERMARK_RE),
        "has_mention": raw.str.contains(r"(?<!\w)@\w+"),
        "text_label_conflict_group": val["review_flags"].str.contains("text_label_conflict"),
        "score_best": next(r.val_scores for r in results if r.experiment_id == best_id).round(4),
    })
    table["review_flag"] = np.where(
        table["models_wrong_of_6"] == 6, "all_text_models_wrong: possible image-dependent meaning, ambiguity or label noise",
        np.where(~table["has_text"], "no_text: text model cannot decide", ""))
    return table[table["error_type"] != "correct"].reset_index(drop=True)


def error_category_rates(val: pd.DataFrame, errors: pd.DataFrame, n_val: int) -> pd.DataFrame:
    """Error rate of the best model within diagnostic sample categories."""
    all_flags = error_table_flags(val)
    wrong = set(errors["sample_id"])
    rows = []
    for name, mask in all_flags.items():
        ids = set(val.loc[mask.to_numpy(), "sample_id"])
        if ids:
            rows.append({"category": name, "samples": len(ids), "errors": len(ids & wrong),
                         "error_rate": round(len(ids & wrong) / len(ids), 4)})
    rows.append({"category": "ALL", "samples": n_val, "errors": len(wrong), "error_rate": round(len(wrong) / n_val, 4)})
    return pd.DataFrame(rows)


def error_table_flags(val: pd.DataFrame) -> dict:
    raw = val["text_raw"]
    words = raw.str.split().str.len()
    return {
        "no_text": raw.str.strip() == "",
        "very_short_1_3_words": words.between(1, 3),
        "long_20plus_words": words >= 20,
        "has_negation": pd.Series([hinglish.count_negations(t) > 0 for t in raw], index=val.index),
        "has_devanagari": raw.str.contains(r"[ऀ-ॿ]"),
        "has_hinglish_rule_variant": pd.Series([bool(hinglish.load_rules().pattern.search(t)) for t in val["text_basic"]],
                                               index=val.index),
        "has_watermark": raw.str.contains(WATERMARK_RE),
        "has_mention": raw.str.contains(r"(?<!\w)@\w+"),
    }
