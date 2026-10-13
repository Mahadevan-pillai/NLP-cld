"""Task 10 report: image-model figures, unimodal comparison, error analysis, diagnostics, registry.

Run after scripts/train_image_models.py train:
    python scripts/report_image_models.py

VALIDATION ONLY. Task 8 / Task 9 results are read, never rewritten.
Text columns are used ONLY to describe errors (flags), never as model input.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.data.eda_plots import CLASS_COLORS, INK_2, _finish, plt  # noqa: E402
from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.evaluation.plots import confusion_matrix_figure  # noqa: E402
from src.models.image_encoder import load_cached_embeddings  # noqa: E402
from src.preprocessing.image import IMAGE_PREPROCESSING_VERSION  # noqa: E402
from src.training.text_baselines import WATERMARK_RE, load_development_data  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

METRICS = REPO_ROOT / "results" / "metrics" / "image_models"
TABLES = REPO_ROOT / "results" / "tables" / "image_models"
FIGURES = REPO_ROOT / "results" / "figures" / "image_models"
REGISTRY = REPO_ROOT / "results" / "metrics" / "image_model_registry.json"
T8_RESULTS = REPO_ROOT / "results" / "tables" / "text_baselines" / "baseline_results.csv"
T9_METRICS = REPO_ROOT / "results" / "metrics" / "text_models"
SOURCE_RES_MIN_TRAIN, SOURCE_RES_PURITY = 20, 0.8      # "source resolution": >= 20 train images, >= 80% one class


def results() -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS.glob("[ID]0[0-9].json"))}


def predictions(exp: str, folder: Path, order) -> pd.DataFrame:
    return pd.read_csv(folder / f"{exp}_validation_predictions.csv").set_index("sample_id").loc[order]


def image_sizes(df) -> pd.DataFrame:
    rows = []
    for p in df["image_path"]:
        with Image.open(REPO_ROOT / p) as im:                    # header only
            rows.append(im.size)
    return pd.DataFrame(rows, columns=["width", "height"], index=df.index)


def unimodal_comparison(res: dict, best_image: str) -> pd.DataFrame:
    b = pd.read_csv(T8_RESULTS).set_index("experiment_id").loc["B03"]
    t = json.loads((T9_METRICS / "T05.json").read_text(encoding="utf-8"))
    i = res[best_image]
    rows = [
        {"model": "B03 TF-IDF + Logistic Regression", "modality": "text (OCR caption)",
         "preprocessing_or_encoder": "HINGLISH_NORMALIZED text; TF-IDF word 1-2-grams", "validation_macro_f1": b.macro_f1,
         "macro_f1_ci95": f"{b.macro_f1_ci95_low:.4f}-{b.macro_f1_ci95_high:.4f}", "validation_accuracy": b.accuracy,
         "training_time_s": b.training_time_s, "validation_inference_time_s": b.inference_time_s,
         "model_size_bytes": int(b.artifact_bytes)},
        {"model": "T05 fine-tuned MuRIL", "modality": "text (OCR caption)",
         "preprocessing_or_encoder": f"HINGLISH_NORMALIZED text; {t['encoder']}@{t['encoder_revision'][:8]}",
         "validation_macro_f1": t["metrics"]["macro_f1"],
         "macro_f1_ci95": f"{t['macro_f1_ci95'][0]:.4f}-{t['macro_f1_ci95'][1]:.4f}",
         "validation_accuracy": t["metrics"]["accuracy"], "training_time_s": round(t["train_seconds"], 1),
         "validation_inference_time_s": round(t["inference_seconds_validation"], 2), "model_size_bytes": t["artifact_bytes"]},
        {"model": f"{best_image} {i['model']}", "modality": "image",
         "preprocessing_or_encoder": f"{i['encoder']}@{i['encoder_revision'][:8]}; 224x224 stretch",
         "validation_macro_f1": i["metrics"]["macro_f1"],
         "macro_f1_ci95": f"{i['macro_f1_ci95'][0]:.4f}-{i['macro_f1_ci95'][1]:.4f}",
         "validation_accuracy": i["metrics"]["accuracy"],
         "training_time_s": round(i["train_seconds"] + i["feature_extraction_seconds_train_val"], 1),
         "validation_inference_time_s": "embedding ~0.16 s/image + classifier",
         "model_size_bytes": i["artifact_bytes"]},
    ]
    return pd.DataFrame(rows)


def nearest_train(x_tr, x_va, y_tr):
    """Cosine similarity of each validation image to its nearest TRAIN image (embedding space)."""
    a = x_tr / np.linalg.norm(x_tr, axis=1, keepdims=True)
    b = x_va / np.linalg.norm(x_va, axis=1, keepdims=True)
    sim = b @ a.T
    idx = sim.argmax(1)
    return sim[np.arange(len(b)), idx], y_tr[idx]


def pca_figure(x_tr, x_va, y_va, names, path):
    pca = PCA(n_components=2, random_state=0).fit(x_tr)            # fitted on TRAIN, validation projected
    z = pca.transform(x_va)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for label in (0, 1):
        m = y_va == label
        ax.scatter(z[m, 0], z[m, 1], s=9, alpha=0.55, color=CLASS_COLORS[label], label=f"{names[label].replace('_', '-')} (n={m.sum()})")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% of train variance)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% of train variance)")
    ax.set_title("SigLIP image embeddings, validation set\n(PCA fitted on train; 2 of 768 dimensions)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    fig.text(0.01, 0.01, "Descriptive only: overlap in 2-D does not mean the classes are inseparable in 768-D.",
             fontsize=7, color=INK_2)
    return _finish(fig, [ax], path), ev


def main() -> int:
    config = load_config()
    names = class_names(config)
    for d in (TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    res = results()
    models = [e for e in res if e.startswith("I")]
    if not models:
        print("No Task 10 results found.")
        return 1
    train, val = load_development_data(config)                     # validation only; test never loaded
    y_tr, y_va = train["target_label"].astype(int).to_numpy(), val["target_label"].astype(int).to_numpy()

    for exp in models:
        m = res[exp]["metrics"]
        confusion_matrix_figure(m["tn"], m["fp"], m["fn"], m["tp"], names,
                                f"{exp}: {res[exp]['model']}\nimage only (no text)", FIGURES / f"{exp}_confusion_matrix.png")
    best = max(models, key=lambda e: res[e]["metrics"]["macro_f1"])
    unimodal_comparison(res, best).to_csv(REPO_ROOT / "results" / "tables" / "model_comparison_unimodal.csv",
                                          index=False, lineterminator="\n")

    # ---- error analysis for the best image model (flags only; no text stored)
    x_tr = load_cached_embeddings(config, "train", train["sample_id"])
    x_va = load_cached_embeddings(config, "validation", val["sample_id"])
    sim, nn_label = nearest_train(x_tr, x_va, y_tr)
    pred = predictions(best, METRICS, val["sample_id"])
    t05 = predictions("T05", T9_METRICS, val["sample_id"])["pred"].to_numpy()
    b03_wrong = set(pd.read_csv(REPO_ROOT / "results/tables/text_baselines/validation_errors.csv")["sample_id"])
    size_tr, size_va = image_sizes(train), image_sizes(val)
    res_tr = size_tr["width"].astype(str) + "x" + size_tr["height"].astype(str)
    res_va = size_va["width"].astype(str) + "x" + size_va["height"].astype(str)
    stats = pd.DataFrame({"res": res_tr.values, "y": y_tr}).groupby("res")["y"].agg(["size", "mean"])
    source = stats[(stats["size"] >= SOURCE_RES_MIN_TRAIN) &
                   ((stats["mean"] >= SOURCE_RES_PURITY) | (stats["mean"] <= 1 - SOURCE_RES_PURITY))]
    words = val["text_raw"].str.split().str.len()
    groups = val["group_id"].map(pd.concat([train, val])["group_id"].value_counts())
    p = pred["pred"].to_numpy()
    table = pd.DataFrame({
        "sample_id": val["sample_id"].values, "true_label": y_va, f"pred_{best}": p,
        f"score_{best}": pred["score"].values,
        "error_type": np.where(p == y_va, "correct", np.where(p == 1, "false_positive", "false_negative")),
        "T05_text_also_wrong": t05 != y_va, "B03_text_also_wrong": val["sample_id"].isin(b03_wrong).values,
        "width": size_va["width"].values, "height": size_va["height"].values,
        "source_resolution_in_train": res_va.isin(source.index).values,
        "low_resolution_min_side_lt_400": (size_va[["width", "height"]].min(axis=1) < 400).values,
        "extreme_aspect_lt_0_5_or_gt_2": ((size_va["width"] / size_va["height"]).pipe(lambda r: (r < 0.5) | (r > 2))).values,
        "caption_empty": (val["text_raw"].str.strip() == "").values,
        "caption_20plus_words": (words >= 20).values,
        "has_watermark_caption": val["text_raw"].str.contains(WATERMARK_RE).values,
        "in_multi_sample_leakage_group": (groups > 1).values,
        "nearest_train_cosine": np.round(sim, 4), "nearest_train_label": nn_label,
        "nearest_train_same_label": nn_label == y_va,
    })
    errors = table[table["error_type"] != "correct"].reset_index(drop=True)
    errors.to_csv(TABLES / "validation_errors.csv", index=False, lineterminator="\n")

    flag_cols = ["source_resolution_in_train", "low_resolution_min_side_lt_400", "extreme_aspect_lt_0_5_or_gt_2",
                 "caption_empty", "caption_20plus_words", "has_watermark_caption", "in_multi_sample_leakage_group"]
    rate_rows = [{"category": "ALL", "samples": len(table), "errors": len(errors),
                  "error_rate": round(len(errors) / len(table), 4)}]
    for c in flag_cols:
        sub = table[table[c]]
        rate_rows.append({"category": c, "samples": len(sub), "errors": int((sub["error_type"] != "correct").sum()),
                          "error_rate": round(float((sub["error_type"] != "correct").mean()), 4) if len(sub) else None})
    pd.DataFrame(rate_rows).to_csv(TABLES / "error_category_rates.csv", index=False, lineterminator="\n")

    # ---- shortcut diagnostic: performance inside vs outside class-correlated "source" resolutions
    shortcut = []
    for name, mask in (("source resolution (train-defined)", table["source_resolution_in_train"].to_numpy()),
                       ("other resolutions", ~table["source_resolution_in_train"].to_numpy())):
        row = {"group": name, "samples": int(mask.sum()), "share_misogynistic": round(float(y_va[mask].mean()), 4)}
        for e in models + ["D01"]:
            pe = predictions(e, METRICS, val["sample_id"])["pred"].to_numpy()
            row[f"macro_f1_{e}"] = classification_metrics(y_va[mask], pe[mask], names)["macro_f1"]
        row["macro_f1_T05_text"] = classification_metrics(y_va[mask], t05[mask], names)["macro_f1"]
        shortcut.append(row)
    pd.DataFrame(shortcut).to_csv(TABLES / "resolution_shortcut_check.csv", index=False, lineterminator="\n")
    source.reset_index().rename(columns={"size": "train_samples", "mean": "train_share_misogynistic"}) \
        .to_csv(TABLES / "source_resolutions_train.csv", index=False, lineterminator="\n")

    # ---- image vs text complementarity (descriptive; NOT a multimodal model)
    img_ok, txt_ok = p == y_va, t05 == y_va
    comp = pd.DataFrame([{
        "image_model": best, "text_model": "T05", "both_correct": int((img_ok & txt_ok).sum()),
        "only_image_correct": int((img_ok & ~txt_ok).sum()), "only_text_correct": int((~img_ok & txt_ok).sum()),
        "both_wrong": int((~img_ok & ~txt_ok).sum()), "prediction_agreement": round(float((p == t05).mean()), 4),
        "error_correlation_phi": round(float(np.corrcoef(~img_ok, ~txt_ok)[0, 1]), 4)}])
    comp.to_csv(TABLES / "image_text_complementarity.csv", index=False, lineterminator="\n")

    # ---- descriptive recall of the binary prediction per annotated sub-label (no sub-label model trained)
    sub_rows = []
    sublabels = ("Objectification", "Prejudice", "Humiliation")
    group_masks = {s: val[s].astype(int).to_numpy() == 1 for s in sublabels}
    group_masks["any misogynistic"] = y_va == 1
    for g, mask in group_masks.items():
        row = {"group": g, "validation_samples": int(mask.sum())}
        for e, pe in [(e, predictions(e, METRICS, val["sample_id"])["pred"].to_numpy()) for e in models] + [("T05_text", t05)]:
            row[f"recall_{e}"] = round(float((pe[mask] == 1).mean()), 4)
        sub_rows.append(row)
    pd.DataFrame(sub_rows).to_csv(TABLES / "subcategory_recall.csv", index=False, lineterminator="\n")

    # ---- embedding diagnostics
    bins = pd.cut(sim, [0, 0.8, 0.9, 0.95, 1.0001], labels=["<0.80", "0.80-0.90", "0.90-0.95", ">=0.95"])
    nn = pd.DataFrame({"nearest_train_cosine_bin": bins, "same_label": nn_label == y_va}) \
        .groupby("nearest_train_cosine_bin", observed=False)["same_label"].agg(["size", "mean"]).reset_index() \
        .rename(columns={"size": "validation_samples", "mean": "nearest_train_same_label_rate"})
    nn.to_csv(TABLES / "nearest_neighbour_diagnostic.csv", index=False, lineterminator="\n")
    _, ev = pca_figure(x_tr, x_va, y_va, names, FIGURES / "validation_embeddings_pca.png")

    # ---- registry for predict_image()
    reg = {"task": "misogyny_detection", "modality": "image", "label_map": {"0": names[0], "1": names[1]},
           "note": "Image only (no OCR / text). Scores are uncalibrated; SVM scores are not probabilities.",
           "models": {e: {"kind": "frozen_logreg" if res[e]["classifier"] == "logistic_regression" else "frozen_svm",
                          "artifact": res[e]["artifact"], "encoder": res[e]["encoder"],
                          "encoder_revision": res[e]["encoder_revision"], "embedding": res[e]["embedding"],
                          "embedding_dim": res[e]["embedding_dim"],
                          "image_preprocessing_version": IMAGE_PREPROCESSING_VERSION,
                          "image_preprocessing": res[e]["image_preprocessing"]} for e in models}}
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    print(pd.DataFrame([{"exp": e, "model": r["model"], "macro_f1": r["metrics"]["macro_f1"], "ci": r["macro_f1_ci95"]}
                        for e, r in res.items()]).to_string(index=False))
    print(f"best image model: {best}; errors {len(errors)}; PCA explained variance {np.round(ev, 3).tolist()}")
    print(comp.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
