"""Train and validate the six TF-IDF baselines B01-B06 (Task 8).

Run from the repository root:
    python scripts/train_text_baselines.py

Uses the canonical TRAIN split for fitting and VALIDATION for comparison.
The TEST split is never loaded: TEST SET NOT EVALUATED IN TASK 8.

Outputs:
    models/text/<id>_<classifier>_tfidf_<mode>_v<version>.joblib   (git-ignored)
    results/tables/text_baselines/*.csv      metrics, comparisons, features, errors (no dataset text)
    results/tables/text_baselines/validation_errors_with_text.csv  (git-ignored: contains text)
    results/figures/text_baselines/*.png     validation confusion matrices
    results/metrics/text_baselines_manifest.json
"""

import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.evaluation.plots import confusion_matrix_figure  # noqa: E402
from src.training import text_baselines as tb  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402
from src.utils.io import write_json_if_changed  # noqa: E402

MODE_LABEL = {"raw": "RAW", "basic": "BASIC", "hinglish_normalized": "HINGLISH_NORMALIZED"}
CLF_LABEL = {"logistic_regression": "TF-IDF + Logistic Regression", "linear_svm": "TF-IDF + Linear SVM"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = load_config()
    base = config["text_model"]["baselines"]
    names = class_names(config)
    seed = config["random_seed"]
    tables = REPO_ROOT / base["results_dir"]
    figures = REPO_ROOT / base["figures_dir"]
    models_dir = REPO_ROOT / base["artifact_dir"]
    for d in (tables, figures, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    train, val = tb.load_development_data(config)          # TEST split is not loaded
    print(f"Train {len(train):,} / validation {len(val):,} (canonical Task 7 split). TEST NOT LOADED.")

    results, rows, per_class, artifacts, warn_log, checks = [], [], [], {}, {}, {}
    for exp_id, (clf, mode) in base["experiments"].items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = tb.run_experiment(exp_id, clf, mode, train, val, config, names)
        warn_log[exp_id] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
        checks[f"{exp_id}_preprocessing_matches_task6"] = (
            tb.check_preprocessing_matches_task6(r.pipeline, train, mode)
            and tb.check_preprocessing_matches_task6(r.pipeline, val, mode))

        # save, reload, reproduce validation predictions
        path = models_dir / f"{exp_id}_{clf}_tfidf_{mode}_v{base['version']}.joblib"
        joblib.dump(r.pipeline, path)
        reloaded = joblib.load(path)
        checks[f"{exp_id}_reload_reproduces_predictions"] = bool(
            np.array_equal(reloaded.predict(val[tb.INPUT_COLUMN].tolist()), r.val_pred))
        artifacts[exp_id] = {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size,
                             "sha256": sha256(path)}
        results.append(r)

        m = r.metrics
        rows.append({
            "experiment_id": exp_id, "model": CLF_LABEL[clf], "preprocessing": MODE_LABEL[mode],
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"], "macro_f1_ci95_low": r.macro_f1_ci[0], "macro_f1_ci95_high": r.macro_f1_ci[1],
            "weighted_f1": m["weighted_f1"],
            "macro_f1_with_text_only": r.subsets["with_text"]["macro_f1"],
            "macro_f1_without_watermark": r.subsets["without_watermark"]["macro_f1"],
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "training_time_s": round(r.train_seconds, 3), "inference_time_s": round(r.predict_seconds, 3),
            "n_tfidf_features": r.n_features, "artifact_bytes": artifacts[exp_id]["bytes"],
            "converged": r.converged,
            "tfidf_settings": json.dumps(base["tfidf"], ensure_ascii=False),
            "classifier_settings": json.dumps(base[clf]),
            "random_seed": seed,
        })
        for label, name in names.items():
            per_class.append({"experiment_id": exp_id, "class": name, "precision": m[f"precision_{name}"],
                              "recall": m[f"recall_{name}"], "f1": m[f"f1_{name}"], "support": m[f"support_{name}"]})
        confusion_matrix_figure(m["tn"], m["fp"], m["fn"], m["tp"], names,
                                f"{exp_id}: {CLF_LABEL[clf]}\ntext variant: {MODE_LABEL[mode]}",
                                figures / f"{exp_id}_confusion_matrix.png")
        print(f"  {exp_id} {clf:20s} {mode:20s} macro-F1 {m['macro_f1']:.4f} "
              f"[{r.macro_f1_ci[0]:.3f}, {r.macro_f1_ci[1]:.3f}]  acc {m['accuracy']:.4f}  "
              f"features {r.n_features:,}  train {r.train_seconds:.2f}s  converged {r.converged}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(tables / "baseline_results.csv", index=False, lineterminator="\n")
    pd.DataFrame(per_class).to_csv(tables / "per_class_metrics.csv", index=False, lineterminator="\n")

    # preprocessing comparison (validation macro-F1)
    comp = []
    for clf in ("logistic_regression", "linear_svm"):
        sub = results_df[results_df["model"] == CLF_LABEL[clf]].set_index("preprocessing")["macro_f1"]
        comp.append({"model": CLF_LABEL[clf], "raw": sub["RAW"], "basic": sub["BASIC"],
                     "hinglish_normalized": sub["HINGLISH_NORMALIZED"],
                     "basic_minus_raw": round(sub["BASIC"] - sub["RAW"], 4),
                     "hinglish_minus_basic": round(sub["HINGLISH_NORMALIZED"] - sub["BASIC"], 4),
                     "hinglish_minus_raw": round(sub["HINGLISH_NORMALIZED"] - sub["RAW"], 4)})
    pd.DataFrame(comp).to_csv(tables / "preprocessing_comparison.csv", index=False, lineterminator="\n")

    # feature inspection (all six models)
    features = pd.concat([tb.top_features(r, names) for r in results], ignore_index=True)
    features.to_csv(tables / "top_features.csv", index=False, lineterminator="\n")

    # error analysis for the best validation model
    best = max(results, key=lambda r: (r.metrics["macro_f1"], -list(base["experiments"]).index(r.experiment_id)))
    errors = tb.error_table(val, results, best.experiment_id)
    errors.to_csv(tables / "validation_errors.csv", index=False, lineterminator="\n")
    errors.merge(val[["sample_id", "text_raw", "text_hinglish_normalized"]], on="sample_id").to_csv(
        tables / "validation_errors_with_text.csv", index=False, lineterminator="\n")   # git-ignored
    rates = tb.error_category_rates(val, errors, len(val))
    rates.to_csv(tables / "error_category_rates.csv", index=False, lineterminator="\n")

    manifest = {
        "task": "Task 8 - TF-IDF baselines", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_set": "NOT EVALUATED IN TASK 8 (test split never loaded)",
        "split": {"assignments_sha256": json.loads((REPO_ROOT / config["splitting"]["manifest_file"])
                                                   .read_text(encoding="utf-8"))["assignments_sha256"],
                  "train": len(train), "validation": len(val)},
        "development_metric": config["evaluation"]["primary_metric"],
        "baseline_version": base["version"], "random_seed": seed,
        "settings": {"tfidf": base["tfidf"], "logistic_regression": base["logistic_regression"],
                     "linear_svm": base["linear_svm"]},
        "best_validation_experiment": best.experiment_id,
        "results": rows, "artifacts": artifacts, "warnings": warn_log, "checks": checks,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "joblib": joblib.__version__},
    }
    write_json_if_changed(REPO_ROOT / "results" / "metrics" / "text_baselines_manifest.json", manifest)

    print(f"Best validation macro-F1: {best.experiment_id} ({best.metrics['macro_f1']:.4f})")
    print(f"Warnings: { {k: v for k, v in warn_log.items() if v} or 'none'}")
    failed = [k for k, v in checks.items() if not v]
    if failed:
        print(f"Checks FAILED: {failed}")
        return 1
    print(f"All {len(checks)} checks passed (preprocessing == Task 6 columns; reload reproduces predictions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
