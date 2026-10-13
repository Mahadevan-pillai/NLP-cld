"""Task 11 report: modality ablation, main comparison, gains, disagreements, figures, registry.

Run after scripts/train_multimodal.py train:
    python scripts/report_multimodal.py

VALIDATION ONLY. Task 8/9/10 results are read, never rewritten. Every comparison uses
the SAME validation samples (joined on sample_id); subsets are stated explicitly.
The MAIN multimodal model is M02, fixed in advance by the task design (not chosen by score).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.data.eda_plots import CLASS_COLORS, SINGLE, INK_2, _finish, plt  # noqa: E402
from src.evaluation.metrics import bootstrap_macro_f1, classification_metrics  # noqa: E402
from src.evaluation.plots import confusion_matrix_figure  # noqa: E402
from src.training.text_baselines import load_development_data, subset_masks  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

METRICS = REPO_ROOT / "results" / "metrics"
TABLES = REPO_ROOT / "results" / "tables" / "multimodal"
FIGURES = REPO_ROOT / "results" / "figures" / "multimodal"
REGISTRY = METRICS / "multimodal_model_registry.json"
MAIN, BASELINE, CONTROL = "M02", "M01", "M03"
TEXT_BEST, IMAGE_BEST = "T05", "I01"
MODELS = {   # id: (folder with <id>_validation_predictions.csv, modality, short description)
    "B03": ("text_baselines_from_errors", "text", "TF-IDF + LR (HINGLISH_NORMALIZED)"),
    "T03": ("text_models", "text", "frozen MuRIL + LR (HINGLISH_NORMALIZED)"),
    "T05": ("text_models", "text", "fine-tuned MuRIL (HINGLISH_NORMALIZED)"),
    "I01": ("image_models", "image", "frozen SigLIP + LR"),
    "M01": ("multimodal", "image+text", "concat(T05 pooled, SigLIP) + LR"),
    "M02": ("multimodal", "image+text", "fusion MLP (T05 pooled + SigLIP)"),
    "M03": ("multimodal", "image+text", "fusion MLP (frozen MuRIL + SigLIP) - control"),
}


def b03_predictions(val: pd.DataFrame) -> np.ndarray:
    """Task 8 stored only B03's errors; correct predictions are the true label (validation only)."""
    wrong = set(pd.read_csv(REPO_ROOT / "results/tables/text_baselines/validation_errors.csv")["sample_id"])
    y = val["target_label"].astype(int).to_numpy()
    return np.where(val["sample_id"].isin(wrong).to_numpy(), 1 - y, y)


def all_predictions(val: pd.DataFrame) -> dict[str, np.ndarray]:
    preds = {}
    for exp, (folder, _, _) in MODELS.items():
        if exp == "B03":
            preds[exp] = b03_predictions(val)
            continue
        p = pd.read_csv(METRICS / folder / f"{exp}_validation_predictions.csv").set_index("sample_id")
        if set(p.index) != set(val["sample_id"]):
            raise RuntimeError(f"{exp}: validation sample ids differ")
        p = p.loc[val["sample_id"]]                                    # aligned by sample_id
        if not (p["true_label"].values == val["target_label"].astype(int).values).all():
            raise RuntimeError(f"{exp}: labels differ")
        preds[exp] = p["pred"].to_numpy()
    return preds


def result_json(exp: str) -> dict:
    folder = {"T03": "text_models", "T05": "text_models", "I01": "image_models"}.get(exp, "multimodal")
    return json.loads((METRICS / folder / f"{exp}.json").read_text(encoding="utf-8"))


def training_curves(exp: str):
    h = pd.read_csv(TABLES / f"{exp}_training_history.csv")
    r = result_json(exp)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(h["epoch"], h["train_loss"], marker="o", ms=3, color=CLASS_COLORS[0], linewidth=2, label="train loss")
    axes[0].plot(h["epoch"], h["val_loss"], marker="o", ms=3, color=CLASS_COLORS[1], linewidth=2, label="validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(h["epoch"], h["val_macro_f1"], marker="o", ms=3, color=SINGLE, linewidth=2)
    best = h["val_macro_f1"].idxmax()
    axes[1].scatter([h.loc[best, "epoch"]], [h.loc[best, "val_macro_f1"]], s=140, facecolors="none", edgecolors=INK_2,
                    linewidths=1.5, zorder=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro-F1")
    # label in the title (not inside the axes) so it can never cover a data point
    axes[1].set_title(f"Validation macro-F1\nselected: epoch {int(h.loc[best, 'epoch'])} ({h.loc[best, 'val_macro_f1']:.4f}, ringed)")
    fig.suptitle(f"{exp}: {r['model']}\n(validation used for early stopping / checkpoint selection)",
                 fontsize=10, fontweight="bold")
    return _finish(fig, list(axes), FIGURES / f"{exp}_training_curves.png")


def main() -> int:
    config = load_config()
    names = class_names(config)
    for d in (TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    _, val = load_development_data(config)                              # validation only; test never loaded
    y = val["target_label"].astype(int).to_numpy()
    preds = all_predictions(val)
    masks = subset_masks(val)
    populations = {"all validation (754)": masks["all"], "caption not empty": masks["with_text"],
                   "without doublemean watermark": masks["without_watermark"]}

    # ---- 11.10 modality ablation (identical populations)
    rows = []
    for pop, mask in populations.items():
        for exp, (_, modality, desc) in MODELS.items():
            m = classification_metrics(y[mask], preds[exp][mask], names)
            ci = bootstrap_macro_f1(y[mask], preds[exp][mask], 1000, config["random_seed"])
            rows.append({"population": pop, "n": int(mask.sum()), "experiment_id": exp, "modality": modality, "model": desc,
                         "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                         "macro_f1": m["macro_f1"], "macro_f1_ci95": f"{ci[0]:.4f}-{ci[1]:.4f}",
                         "weighted_f1": m["weighted_f1"]})
    ablation = pd.DataFrame(rows)
    ablation.to_csv(TABLES / "modality_ablation.csv", index=False, lineterminator="\n")

    # ---- 11.14 main comparison
    b03 = pd.read_csv(REPO_ROOT / "results/tables/text_baselines/baseline_results.csv").set_index("experiment_id").loc["B03"]
    timing = {"B03": (b03.training_time_s, f"{b03.inference_time_s} s (754 texts)")}
    for exp in ("T05", "I01", "M01", "M02", "M03"):
        r = result_json(exp)
        if exp == "T05":
            timing[exp] = (round(r["train_seconds"], 1), f"{r['inference_seconds_validation']:.1f} s (754 texts, encoder incl.)")
        elif exp == "I01":
            timing[exp] = (round(r["train_seconds"] + r["feature_extraction_seconds_train_val"], 1),
                           "~0.21 s/image end-to-end")
        else:
            timing[exp] = (r["train_seconds"], f"{r['inference_seconds_classifier_only']} s classifier only "
                                               "(+ T05 text and SigLIP image encoders)")
    main_rows = []
    for exp, role in (("B03", "best traditional text"), ("T05", "best pretrained text"), ("I01", "best image-only"),
                      ("M01", "multimodal baseline"), ("M02", "MAIN multimodal"), ("M03", "multimodal control (frozen text)")):
        m = classification_metrics(y, preds[exp], names)
        main_rows.append({"role": role, "experiment_id": exp, "model": MODELS[exp][2], "modality": MODELS[exp][1],
                          "validation_samples": len(y), "accuracy": m["accuracy"], "precision": m["precision"],
                          "recall": m["recall"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                          "training_time_s": timing[exp][0], "inference_time": timing[exp][1]})
    main_table = pd.DataFrame(main_rows)
    for col in ("accuracy", "precision", "recall", "macro_f1", "weighted_f1"):
        main_table[f"best_{col}"] = main_table[col] == main_table[col].max()
    main_table.to_csv(TABLES / "main_comparison.csv", index=False, lineterminator="\n")

    # ---- 11.15 gains (absolute macro-F1 differences, same samples; no significance claimed)
    f1 = {e: classification_metrics(y, p, names)["macro_f1"] for e, p in preds.items()}
    gains = []
    for exp in ("M01", "M02", "M03"):
        matched_text = "T05" if exp != "M03" else "T03"
        gains.append({"multimodal": exp, "macro_f1": f1[exp],
                      "minus_best_text_T05": round(f1[exp] - f1["T05"], 4),
                      "minus_best_image_I01": round(f1[exp] - f1["I01"], 4),
                      "matched_text_model": matched_text, "minus_matched_text": round(f1[exp] - f1[matched_text], 4)})
    pd.DataFrame(gains).to_csv(TABLES / "multimodal_gain.csv", index=False, lineterminator="\n")

    # ---- 11.16 disagreements (text = T05, image = I01, multimodal = MAIN)
    t_ok, i_ok, m_ok = preds[TEXT_BEST] == y, preds[IMAGE_BEST] == y, preds[MAIN] == y
    cats = {
        "A_text_correct_image_wrong_mm_correct": t_ok & ~i_ok & m_ok,
        "B_image_correct_text_wrong_mm_correct": i_ok & ~t_ok & m_ok,
        "C_text_wrong_image_wrong_mm_correct": ~t_ok & ~i_ok & m_ok,
        "D_text_correct_image_correct_mm_wrong": t_ok & i_ok & ~m_ok,
        "E_text_and_image_predictions_disagree": preds[TEXT_BEST] != preds[IMAGE_BEST],
        "F_all_three_wrong": ~t_ok & ~i_ok & ~m_ok,
    }
    per_sample = pd.DataFrame({"sample_id": val["sample_id"].values, "true_label": y,
                               f"pred_text_{TEXT_BEST}": preds[TEXT_BEST], f"pred_image_{IMAGE_BEST}": preds[IMAGE_BEST],
                               f"pred_multimodal_{MAIN}": preds[MAIN], f"pred_multimodal_{BASELINE}": preds[BASELINE],
                               f"pred_multimodal_{CONTROL}": preds[CONTROL],
                               "caption_empty": ~masks["with_text"], **{k: v for k, v in cats.items()}})
    per_sample.to_csv(TABLES / "modality_disagreements.csv", index=False, lineterminator="\n")
    summary = []
    for k, v in cats.items():
        row = {"category": k, "samples": int(v.sum()), "share_of_validation": round(float(v.mean()), 4)}
        if k.startswith("E_"):
            row["mm_correct_when_disagree"] = int((v & m_ok).sum())
            row["mm_follows_text"] = int((v & (preds[MAIN] == preds[TEXT_BEST])).sum())
            row["mm_follows_image"] = int((v & (preds[MAIN] == preds[IMAGE_BEST])).sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(TABLES / "disagreement_summary.csv", index=False, lineterminator="\n")

    # ---- descriptive sub-label recall (no sub-label model trained)
    sub = []
    for g in ("Objectification", "Prejudice", "Humiliation"):
        mask = val[g].astype(int).to_numpy() == 1
        sub.append({"group": g, "validation_samples": int(mask.sum()),
                    **{f"recall_{e}": round(float((preds[e][mask] == 1).mean()), 4) for e in ("T05", "I01", "M01", "M02", "M03")}})
    pd.DataFrame(sub).to_csv(TABLES / "subcategory_recall.csv", index=False, lineterminator="\n")

    # ---- figures
    for exp in ("M01", "M02", "M03"):
        m = result_json(exp)["metrics"]
        confusion_matrix_figure(m["tn"], m["fp"], m["fn"], m["tp"], names, f"{exp}: {MODELS[exp][2]}\nimage + text",
                                FIGURES / f"{exp}_confusion_matrix.png")
    for exp in ("M02", "M03"):
        training_curves(exp)

    # ---- registry for predict_multimodal()
    tcfg, icfg = config["text_model"]["transformer"], config["image_model"]["encoder"]
    reps = config["multimodal_model"]["text_representations"]
    reg = {"task": "misogyny_detection", "modality": "image+text", "label_map": {"0": names[0], "1": names[1]},
           "note": "Needs image AND caption (no OCR). Scores are uncalibrated.", "main_model": MAIN, "models": {}}
    for exp in ("M01", "M02", "M03"):
        r = result_json(exp)
        rep = r["text_representation"]
        reg["models"][exp] = {
            "kind": "concat_logreg" if r["classifier"] == "logistic_regression" else "fusion_mlp",
            "artifact": r["artifact"], "text_representation": rep,
            "text_source": reps[rep]["artifact"] if rep == "finetuned_pooled" else tcfg["checkpoint"],
            "text_encoder_revision": tcfg["revision"], "text_preprocessing_mode": r["text_preprocessing_mode"],
            "text_preprocessing_version": r["text_preprocessing_version"], "max_length": tcfg["max_length"],
            "image_encoder": icfg["checkpoint"], "image_encoder_revision": icfg["revision"],
            "image_preprocessing_version": r["image_preprocessing_version"]}
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    print(main_table[["role", "experiment_id", "macro_f1", "accuracy"]].to_string(index=False))
    print(pd.DataFrame(gains).to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
