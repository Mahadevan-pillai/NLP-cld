"""Task 9 report: compare text models on VALIDATION, figures, errors, model registry.

Run after scripts/train_text_transformers.py:
    python scripts/report_text_models.py

Reads Task 8 results READ-ONLY (never rewrites them) and Task 9 results from
results/metrics/text_models/. TEST SET NOT EVALUATED.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.data.eda_plots import CLASS_COLORS, INK_2, SINGLE, _finish, plt  # noqa: E402
from src.evaluation.plots import confusion_matrix_figure  # noqa: E402
from src.training.text_baselines import error_category_rates, error_table_flags, load_development_data  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

T8 = REPO_ROOT / "results" / "tables" / "text_baselines" / "baseline_results.csv"
METRICS = REPO_ROOT / "results" / "metrics" / "text_models"
TABLES = REPO_ROOT / "results" / "tables" / "text_models"
FIGURES = REPO_ROOT / "results" / "figures" / "text_models"
REGISTRY = REPO_ROOT / "results" / "metrics" / "text_model_registry.json"


def task9_results() -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS.glob("T0[0-9].json"))}


def comparison(t9: dict) -> pd.DataFrame:
    rows = []
    for _, r in pd.read_csv(T8).iterrows():
        rows.append({"experiment_id": r.experiment_id, "task_id": "Task 8", "model": r.model,
                     "preprocessing": r.preprocessing, "task": "misogyny_detection (binary)",
                     "accuracy": r.accuracy, "precision": r.precision, "recall": r.recall, "macro_f1": r.macro_f1,
                     "macro_f1_ci95_low": r.macro_f1_ci95_low, "macro_f1_ci95_high": r.macro_f1_ci95_high,
                     "weighted_f1": r.weighted_f1, "macro_f1_with_text_only": r.macro_f1_with_text_only,
                     "macro_f1_without_watermark": r.macro_f1_without_watermark,
                     "training_time_s": r.training_time_s, "inference_time_s": r.inference_time_s,
                     "model_size_bytes": r.artifact_bytes})
    for exp, r in t9.items():
        m = r["metrics"]
        rows.append({"experiment_id": exp, "task_id": "Task 9", "model": r["model"], "preprocessing": r["preprocessing"],
                     "task": r["task"], "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                     "macro_f1": m["macro_f1"], "macro_f1_ci95_low": r["macro_f1_ci95"][0],
                     "macro_f1_ci95_high": r["macro_f1_ci95"][1], "weighted_f1": m["weighted_f1"],
                     "macro_f1_with_text_only": r["subset_macro_f1"]["with_text"],
                     "macro_f1_without_watermark": r["subset_macro_f1"]["without_watermark"],
                     "training_time_s": round(r["train_seconds"] + r.get("embedding_seconds_train_val", 0), 2),
                     "inference_time_s": round(r.get("inference_seconds_validation", r.get("inference_seconds_classifier_only", 0)), 3),
                     "model_size_bytes": r["artifact_bytes"]})
    return pd.DataFrame(rows)


def normalization_table(t9: dict) -> pd.DataFrame:
    f1 = {e: r["metrics"]["macro_f1"] for e, r in t9.items()}
    rows = []
    if {"T01", "T02", "T03"} <= set(f1):
        rows.append({"level": "frozen MuRIL + LR", "raw": f1["T01"], "basic": f1["T02"], "hinglish_normalized": f1["T03"],
                     "hinglish_minus_basic": round(f1["T03"] - f1["T02"], 4), "hinglish_minus_raw": round(f1["T03"] - f1["T01"], 4)})
    if {"T04", "T05"} <= set(f1):
        base = t9["T04"]["preprocessing"]
        rows.append({"level": f"fine-tuned MuRIL ({base} vs HINGLISH)", "comparison_variant": base,
                     "comparison_macro_f1": f1["T04"], "hinglish_normalized": f1["T05"],
                     "hinglish_minus_comparison": round(f1["T05"] - f1["T04"], 4)})
    return pd.DataFrame(rows)


def training_curves(exp: str) -> Path | None:
    path = TABLES / f"{exp}_training_history.csv"
    if not path.exists():
        return None
    h = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(h["epoch"], h["train_loss"], marker="o", color=CLASS_COLORS[0], linewidth=2, label="train loss")
    axes[0].plot(h["epoch"], h["val_loss"], marker="o", color=CLASS_COLORS[1], linewidth=2, label="validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(h["epoch"], h["val_macro_f1"], marker="o", color=SINGLE, linewidth=2)
    best = h["val_macro_f1"].idxmax()
    axes[1].annotate(f"best: epoch {int(h.loc[best, 'epoch'])}\n{h.loc[best, 'val_macro_f1']:.4f}",
                     (h.loc[best, "epoch"], h.loc[best, "val_macro_f1"]), xytext=(8, -24), textcoords="offset points",
                     fontsize=8, color=INK_2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro-F1")
    axes[1].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xticks(h["epoch"])
    fig.suptitle(f"{exp}: fine-tuning MuRIL (validation used for checkpoint selection)", fontsize=10, fontweight="bold")
    return _finish(fig, list(axes), FIGURES / f"{exp}_training_curves.png")


def error_analysis(val: pd.DataFrame, best: str) -> pd.DataFrame:
    y = val["target_label"].astype(int).to_numpy()
    pred = pd.read_csv(METRICS / f"{best}_validation_predictions.csv").set_index("sample_id").loc[val["sample_id"]]
    b03 = pd.read_csv(REPO_ROOT / "results" / "tables" / "text_baselines" / "validation_errors.csv")
    b03_wrong = set(b03["sample_id"])                 # samples the best Task 8 model got wrong
    flags = error_table_flags(val)
    table = pd.DataFrame({
        "sample_id": val["sample_id"].values, "true_label": y, f"pred_{best}": pred["pred"].values,
        f"score_{best}_uncalibrated": pred["score"].values,
        "error_type": ["false_positive" if (p == 1 and t == 0) else "false_negative" if (p == 0 and t == 1) else "correct"
                       for p, t in zip(pred["pred"].values, y)],
        "B03_also_wrong": val["sample_id"].isin(b03_wrong).values,
        **{k: v.values for k, v in flags.items()},
        "text_label_conflict_group": val["review_flags"].str.contains("text_label_conflict").values,
    })
    return table[table["error_type"] != "correct"].reset_index(drop=True)


def registry(t9: dict, config: dict) -> dict:
    tp = config["text_preprocessing"]
    models = {}
    for _, r in pd.read_csv(T8).iterrows():
        kind = "tfidf_logreg" if "Logistic" in r.model else "tfidf_svm"
        mode = {"RAW": "raw", "BASIC": "basic", "HINGLISH_NORMALIZED": "hinglish_normalized"}[r.preprocessing]
        art = json.loads((REPO_ROOT / "results/metrics/text_baselines_manifest.json").read_text(encoding="utf-8"))
        models[r.experiment_id] = {"kind": kind, "artifact": art["artifacts"][r.experiment_id]["path"],
                                   "preprocessing_mode": mode, "preprocessing_version": tp["version"]}
    for exp, r in t9.items():
        entry = {"preprocessing_mode": r["preprocessing_mode"], "preprocessing_version": tp["version"],
                 "max_length": r["max_length"], "encoder": r["encoder"], "encoder_revision": r["encoder_revision"],
                 "artifact": r["artifact"]}
        entry["kind"] = "frozen_logreg" if exp in ("T01", "T02", "T03") else "fine_tuned"
        models[exp] = entry
    return {"task": "misogyny_detection", "label_map": {"0": "non_misogynistic", "1": "misogynistic"},
            "note": "Scores are uncalibrated. Large checkpoints: {MODEL_DIR} = $NLP_CLD_MODEL_DIR.", "models": models}


def main() -> int:
    config = load_config()
    names = class_names(config)
    for d in (TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    t9 = task9_results()
    if not t9:
        print("No Task 9 results found.")
        return 1
    _, val = load_development_data(config)              # validation only
    comp = comparison(t9)
    comp.to_csv(TABLES / "text_model_comparison.csv", index=False, lineterminator="\n")
    normalization_table(t9).to_csv(TABLES / "normalization_comparison.csv", index=False, lineterminator="\n")
    for exp, r in t9.items():
        m = r["metrics"]
        confusion_matrix_figure(m["tn"], m["fp"], m["fn"], m["tp"], names,
                                f"{exp}: {r['model']}\ntext variant: {r['preprocessing']}",
                                FIGURES / f"{exp}_confusion_matrix.png")
        training_curves(exp)
    best = max(t9, key=lambda e: t9[e]["metrics"]["macro_f1"])
    errors = error_analysis(val, best)
    errors.to_csv(TABLES / "validation_errors.csv", index=False, lineterminator="\n")
    errors.merge(val[["sample_id", "text_raw"]], on="sample_id").to_csv(
        TABLES / "validation_errors_with_text.csv", index=False, lineterminator="\n")       # git-ignored
    error_category_rates(val, errors, len(val)).to_csv(TABLES / "error_category_rates.csv", index=False,
                                                       lineterminator="\n")
    REGISTRY.write_text(json.dumps(registry(t9, config), indent=2), encoding="utf-8")
    print(comp[["experiment_id", "model", "preprocessing", "macro_f1", "macro_f1_ci95_low", "macro_f1_ci95_high"]]
          .to_string(index=False))
    print(f"Best Task 9 model on validation: {best} ({t9[best]['metrics']['macro_f1']:.4f}); errors: {len(errors)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
