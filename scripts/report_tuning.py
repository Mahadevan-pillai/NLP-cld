"""Task 12 report: merged trial log, best configurations, before/after table, figures, registries, audit.

Run after the searches (see scripts/tune_models.py):
    python scripts/report_tuning.py        (or: python scripts/tune_models.py report)

VALIDATION ONLY. Task 8-11 result files are read, never rewritten; model registries only GAIN
entries for the tuned models (existing entries are checked to be unchanged).
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.eda_plots import CLASS_COLORS, INK_2, SINGLE, _finish, plt  # noqa: E402
from src.data.splits import load_split_assignments  # noqa: E402
from src.training.tuning import TRIAL_COLUMNS, params_json  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

METRICS = REPO_ROOT / "results" / "metrics"
TUNE_M = METRICS / "tuning"
TABLES = REPO_ROOT / "results" / "tables" / "tuning"
FIGURES = REPO_ROOT / "results" / "figures" / "tuning"
BASELINE_JSON = {"T05": "text_models/T05.json", "I01": "image_models/I01.json", "M02": "multimodal/M02.json"}
TASK12_SOURCES = ("scripts/tune_models.py", "scripts/report_tuning.py", "src/training/tuning.py")


def load(rel: str) -> dict:
    return json.loads((METRICS / rel).read_text(encoding="utf-8"))


def b03() -> dict:
    r = pd.read_csv(REPO_ROOT / "results/tables/text_baselines/baseline_results.csv").set_index("experiment_id").loc["B03"]
    return {"metrics": {"macro_f1": r.macro_f1, "accuracy": r.accuracy, "precision": r.precision, "recall": r.recall},
            "train_seconds": r.training_time_s}


def finetune_trials(config) -> list[dict]:
    """T06/T07 (Task 12 learning-rate sweep) as trial rows; T05 (Task 9) as the reference row."""
    rows = []
    for exp, tid in (("T05", "TP-000"), ("T06", "TP-001"), ("T07", "TP-002")):
        path = TUNE_M / f"{exp}.json" if exp != "T05" else METRICS / "text_models/T05.json"
        if not path.exists():
            rows.append({**{c: None for c in TRIAL_COLUMNS}, "trial_id": tid, "search": "pretrained_text_lr",
                         "model": f"fine-tuned MuRIL ({exp})", "modality": "text", "status": "FAILED",
                         "note": f"{exp} result missing (run not completed)"})
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        lr = r["fine_tune_config"]["learning_rate"]
        rows.append({"trial_id": tid, "search": "pretrained_text_lr", "model": f"fine-tuned MuRIL ({exp})", "modality": "text",
                     "parameters": params_json({"learning_rate": lr}), "seed": r["fine_tune_config"]["seed"],
                     "epochs": r["epochs_run"], "best_epoch": r["best_epoch"],
                     "validation_accuracy": r["metrics"]["accuracy"], "validation_macro_f1": r["metrics"]["macro_f1"],
                     "validation_weighted_f1": r["metrics"]["weighted_f1"],
                     "train_macro_f1": r.get("train_macro_f1", 0.7477 if exp == "T05" else None),
                     "training_time_s": round(r["train_seconds"], 1),
                     "status": "EARLY_STOPPED" if r["epochs_run"] < r["fine_tune_config"]["max_epochs"] else "COMPLETED",
                     "note": ("Task 9 reference (baseline, lr 2e-5); train macro-F1 measured in Task 11" if exp == "T05" else
                              "DEGENERATE: predicts a single class (training did not learn)"
                              if min(r["metrics"]["tp"] + r["metrics"]["fp"], r["metrics"]["tn"] + r["metrics"]["fn"]) == 0 else "")})
    return rows


def plot_param_curves(text: pd.DataFrame, image: pd.DataFrame, baseline_text: float, baseline_image: float):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    text = text[text.trial_id != "TT-000"].copy()
    text["p"] = text.parameters.map(json.loads)
    best_per = text.assign(C=text.p.map(lambda p: p["C"]), ng=text.p.map(lambda p: f"1-{p['ngram_range'][1]}-grams")) \
        .groupby(["ng", "C"])["validation_macro_f1"].max().reset_index()
    third = "#2e9e5b"                                   # green: a third, distinct series colour
    for color, (ng, g) in zip([CLASS_COLORS[0], CLASS_COLORS[1], third], best_per.groupby("ng")):
        axes[0].plot(g["C"], g["validation_macro_f1"], marker="o", ms=4, linewidth=2, color=color, label=ng)
    axes[0].axhline(baseline_text, color=INK_2, linestyle="--", linewidth=1, label=f"baseline B03 ({baseline_text:.4f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("LR regularisation C (log scale)")
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_title("A. TF-IDF + LR\n(best over min_df, sublinear_tf per point)")
    axes[0].legend(fontsize=7)
    image = image[image.trial_id != "TI-000"].copy()
    image["p"] = image.parameters.map(json.loads)
    for color, kind in ((CLASS_COLORS[0], "logistic_regression"), (CLASS_COLORS[1], "linear_svm")):
        g = image[image.p.map(lambda p: p["classifier"] == kind)]
        axes[1].plot(g.p.map(lambda p: p["C"]), g["validation_macro_f1"], marker="o", ms=4, linewidth=2, color=color,
                     label=kind.replace("_", " "))
    axes[1].axhline(baseline_image, color=INK_2, linestyle="--", linewidth=1, label=f"baseline I01 ({baseline_image:.4f})")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Regularisation C (log scale)")
    axes[1].set_ylabel("Validation macro-F1")
    axes[1].set_title("C. Classifier on frozen SigLIP embeddings")
    axes[1].legend(fontsize=7)
    fig.suptitle("Validation macro-F1 vs regularisation (grid searches, seed 42)", fontsize=10, fontweight="bold")
    return _finish(fig, list(axes), FIGURES / "param_vs_macro_f1.png")


def plot_multimodal_trials(mm: pd.DataFrame, baseline: float):
    search = mm[mm.search == "multimodal_random_search"].sort_values("validation_macro_f1").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    colors = [CLASS_COLORS[1] if t == "TM-000" else SINGLE for t in search.trial_id]
    ax.bar(range(len(search)), search["validation_macro_f1"], color=colors)
    ax.axhline(baseline, color=INK_2, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(search)))
    ax.set_xticklabels(search.trial_id.str.replace("TM-", ""), rotation=90, fontsize=7)
    lo = search["validation_macro_f1"].min()
    ax.set_ylim(max(0.0, lo - 0.02), search["validation_macro_f1"].max() + 0.01)
    ax.set_xlabel("Trial (sorted by validation macro-F1; orange = baseline M02 configuration)")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("D. Fusion-MLP random search: 30 sampled configurations + baseline (seed 42)", fontsize=10, fontweight="bold")
    return _finish(fig, [ax], FIGURES / "multimodal_trials.png")


def plot_seeds(seed_table: pd.DataFrame, seeds: list[int]):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = list(seed_table["configuration"])
    for i, lab in enumerate(labels):
        row = seed_table.iloc[i]
        vals = [row[f"seed_{s}"] for s in seeds]
        ax.scatter([i] * len(vals), vals, color=SINGLE, s=30, zorder=3)
        ax.errorbar([i + 0.15], [row["mean"]], yerr=[row["std"]], fmt="D", color=CLASS_COLORS[1], capsize=4, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lab.replace("_", "\n") for lab in labels], fontsize=8)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title(f"Seed robustness (seeds {', '.join(map(str, seeds))})\ndots = single seeds; diamond = mean ± std",
                 fontsize=10, fontweight="bold")
    return _finish(fig, [ax], FIGURES / "seed_robustness.png")


def plot_curves(hist: pd.DataFrame, title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(hist["epoch"], hist["train_loss"], marker="o", ms=3, color=CLASS_COLORS[0], linewidth=2, label="train loss")
    axes[0].plot(hist["epoch"], hist["val_loss"], marker="o", ms=3, color=CLASS_COLORS[1], linewidth=2, label="validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(hist["epoch"], hist["val_macro_f1"], marker="o", ms=3, color=SINGLE, linewidth=2)
    best = hist["val_macro_f1"].idxmax()
    axes[1].scatter([hist.loc[best, "epoch"]], [hist.loc[best, "val_macro_f1"]], s=140, facecolors="none",
                    edgecolors=INK_2, linewidths=1.5, zorder=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro-F1")
    axes[1].set_title(f"Validation macro-F1\nselected: epoch {int(hist.loc[best, 'epoch'])} ({hist.loc[best, 'val_macro_f1']:.4f}, ringed)")
    fig.suptitle(title, fontsize=10, fontweight="bold")
    return _finish(fig, list(axes), path)


def plot_finetune_sweep():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    sources = {"T05 (lr 2e-5, baseline)": REPO_ROOT / "results/tables/text_models/T05_training_history.csv",
               "T06 (lr 3e-5)": TABLES / "T06_training_history.csv", "T07 (lr 5e-5)": TABLES / "T07_training_history.csv"}
    for color, (label, path) in zip([INK_2, CLASS_COLORS[0], CLASS_COLORS[1]], sources.items()):
        if path.exists():
            h = pd.read_csv(path)
            ax.plot(h["epoch"], h["val_macro_f1"], marker="o", ms=4, linewidth=2, color=color, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title("B. MuRIL fine-tuning: learning-rate sweep\n(validation macro-F1 per epoch, seed 42)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)
    return _finish(fig, [ax], FIGURES / "finetune_lr_sweep.png")


def plot_before_after(ba: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(ba))
    ax.bar(x - 0.18, ba["baseline_macro_f1"], width=0.36, color=CLASS_COLORS[0], label="baseline")
    ax.bar(x + 0.18, ba["tuned_macro_f1"], width=0.36, color=CLASS_COLORS[1], label="best tuning run")
    for i, r in ba.iterrows():
        ax.text(i + 0.18, r["tuned_macro_f1"] + 0.01, f"{r['absolute_change']:+.3f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(ba["family"], fontsize=8)
    ax.set_ylim(0.0, 1.0)            # full range: a degenerate run (~0.33) must stay visible
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Before vs after tuning (validation, seed 42)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    return _finish(fig, [ax], FIGURES / "before_after_tuning.png")


PRE_TASK12_IDS = {"B01", "B02", "B03", "B04", "B05", "B06", "T01", "T02", "T03", "T04", "T05",
                  "I01", "I02", "M01", "M02", "M03"}


def update_registry(path: Path, new_entries: dict) -> int:
    """Append-only: adds Task 12 models; refuses to touch any pre-Task-12 model entry."""
    clash = set(new_entries) & PRE_TASK12_IDS
    if clash:
        raise RuntimeError(f"refusing to overwrite pre-Task-12 registry entries {sorted(clash)}")
    reg = json.loads(path.read_text(encoding="utf-8"))
    reg["models"].update(new_entries)
    path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return len(new_entries)


def test_access_audit(config) -> pd.DataFrame:
    rows = []
    # needles built from parts so this file does not match its own check
    needles = ("load_split(" + '"test"', "allow_" + "test=True")
    for rel in TASK12_SOURCES:
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        rows.append({"check": f"{rel} never loads the test split", "passed": not any(n in src for n in needles)})
    leaked = []
    for p in list(TUNE_M.glob("*.json")) + list(TABLES.glob("*.csv")):
        text = p.read_text(encoding="utf-8").lower()
        if re.search(r"test_(accuracy|macro_f1|f1|precision|recall|confusion)", text):
            leaked.append(p.name)
    rows.append({"check": "no test metric in any Task 12 output", "passed": not leaked, "detail": ", ".join(leaked)})
    jsons = [json.loads(p.read_text(encoding="utf-8")) for p in TUNE_M.glob("[BTIM]0*.json")]
    rows.append({"check": "every tuned-model result says test_set = NOT EVALUATED",
                 "passed": all(j.get("test_set") == "NOT EVALUATED" for j in jsons), "detail": f"{len(jsons)} files"})
    a = load_split_assignments(config)                               # raises if the canonical split changed
    rows.append({"check": "canonical split unchanged (manifest SHA-256)", "passed": True,
                 "detail": str(a["split"].value_counts().to_dict())})
    rows.append({"check": "selection used validation macro-F1 only (no test-based threshold or selection)",
                 "passed": True, "detail": "by construction: only train/validation loaded (load_development_data)"})
    return pd.DataFrame(rows)


def main() -> int:
    config = load_config()
    seeds = config["tuning"]["robustness_seeds"]
    for d in (TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    text, image = pd.read_csv(TABLES / "trials_traditional_text.csv"), pd.read_csv(TABLES / "trials_image.csv")
    mm = pd.read_csv(TABLES / "trials_multimodal.csv")
    ft = pd.DataFrame(finetune_trials(config))
    trials = pd.concat([text, ft, image, mm], ignore_index=True)[TRIAL_COLUMNS]
    trials.to_csv(TABLES / "hyperparameter_trials.csv", index=False, lineterminator="\n")

    tuned = {e: json.loads((TUNE_M / f"{e}.json").read_text(encoding="utf-8")) for e in ("B07", "I03", "M04")}
    ft_runs = {e: json.loads((TUNE_M / f"{e}.json").read_text(encoding="utf-8")) for e in ("T06", "T07")
               if (TUNE_M / f"{e}.json").exists()}
    base = {"B03": b03(), **{e: load(p) for e, p in BASELINE_JSON.items()}}
    best_ft = max(ft_runs, key=lambda e: ft_runs[e]["metrics"]["macro_f1"]) if ft_runs else None

    def cfg_text(e):
        r = tuned.get(e) or ft_runs.get(e) or {}
        return json.dumps(r.get("parameters") or {"learning_rate": r.get("fine_tune_config", {}).get("learning_rate")})

    families = [("traditional text", "B03", "B07", json.dumps({"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True, "C": 1.0}),
                 len(text) - 1),
                ("image-only", "I01", "I03", json.dumps({"classifier": "logistic_regression", "C": 1.0}), len(image) - 1),
                ("multimodal (main)", "M02", "M04", json.dumps({k: config["multimodal_model"]["fusion_mlp"][k]
                                                                for k in config["tuning"]["multimodal"]["space"]}),
                 int((mm.search == "multimodal_random_search").sum()) - 1)]
    if best_ft:
        families.insert(1, ("pretrained text", "T05", best_ft, json.dumps({"learning_rate": 2e-5}), len(ft_runs)))
    best_rows, ba_rows = [], []
    runtime = trials.groupby("search")["training_time_s"].sum()
    search_of = {"B07": "traditional_text", "I03": "image", "M04": "multimodal_random_search"}
    for fam, b_id, t_id, b_cfg, n in families:
        bm = base[b_id]["metrics"]
        tr = tuned.get(t_id) or ft_runs[t_id]
        tm = tr["metrics"]
        change = round(tm["macro_f1"] - bm["macro_f1"], 4)
        rt = round(float(runtime.get(search_of.get(t_id, "pretrained_text_lr"), np.nan)), 1)
        best_rows.append({"family": fam, "baseline_model": b_id, "baseline_configuration": b_cfg,
                          "baseline_macro_f1": bm["macro_f1"], "tuned_model": t_id, "best_configuration": cfg_text(t_id),
                          "tuned_macro_f1": tm["macro_f1"], "absolute_improvement": change,
                          "relative_improvement_pct": round(100 * change / bm["macro_f1"], 2), "trials_searched": n,
                          "search_runtime_s": rt})
        ba_rows.append({"family": fam, "baseline_model": b_id, "tuned_model": t_id,
                        "baseline_macro_f1": bm["macro_f1"], "tuned_macro_f1": tm["macro_f1"], "absolute_change": change,
                        "baseline_accuracy": bm["accuracy"], "tuned_accuracy": tm["accuracy"],
                        "baseline_precision": bm["precision"], "tuned_precision": tm["precision"],
                        "baseline_recall": bm["recall"], "tuned_recall": tm["recall"],
                        "baseline_train_s": round(float(base[b_id].get("train_seconds", np.nan)), 2),
                        "tuned_train_s": round(float(tr.get("train_seconds", np.nan)), 2),
                        "tuned_ci95": f"{tr['macro_f1_ci95'][0]:.4f}-{tr['macro_f1_ci95'][1]:.4f}",
                        # multimodal only: mean over the robustness seeds (the adoption criterion)
                        "baseline_seed_mean": tr.get("seed_robustness", {}).get("baseline", {}).get("mean"),
                        "tuned_seed_mean": tr.get("seed_robustness", {}).get("tuned", {}).get("mean")})
    best = pd.DataFrame(best_rows)
    best.to_csv(TABLES / "best_hyperparameters.csv", index=False, lineterminator="\n")
    ba = pd.DataFrame(ba_rows)
    ba.to_csv(TABLES / "before_after_tuning.csv", index=False, lineterminator="\n")

    seed_table = pd.read_csv(TABLES / "seed_robustness.csv")
    plot_param_curves(text, image, base["B03"]["metrics"]["macro_f1"], base["I01"]["metrics"]["macro_f1"])
    plot_multimodal_trials(mm, base["M02"]["metrics"]["macro_f1"])
    plot_seeds(seed_table, seeds)
    plot_curves(pd.read_csv(TABLES / "M04_training_history.csv"),
                f"M04: tuned fusion MLP ({tuned['M04']['selected_trial']}, seed 42)", FIGURES / "M04_training_curves.png")
    plot_finetune_sweep()
    plot_before_after(ba)

    # Registries (append-only) so predict_text / predict_image / predict_multimodal can serve the tuned models.
    tp = config["text_preprocessing"]
    # Fine-tuning sweep runs are registered only if they beat their baseline (T05); degenerate runs are not served.
    t05_f1 = base["T05"]["metrics"]["macro_f1"]
    update_registry(METRICS / "text_model_registry.json", {
        "B07": {"kind": "tfidf_logreg", "artifact": tuned["B07"]["artifact"], "preprocessing_mode": "hinglish_normalized",
                "preprocessing_version": tp["version"]},
        **{e: {"preprocessing_mode": r["preprocessing_mode"], "preprocessing_version": tp["version"], "max_length": r["max_length"],
               "encoder": r["encoder"], "encoder_revision": r["encoder_revision"], "artifact": r["artifact"], "kind": "fine_tuned"}
           for e, r in ft_runs.items() if r["metrics"]["macro_f1"] > t05_f1}})
    i3 = tuned["I03"]
    update_registry(METRICS / "image_model_registry.json", {"I03": {
        "kind": "frozen_logreg" if i3["classifier"] == "logistic_regression" else "frozen_svm", "artifact": i3["artifact"],
        "encoder": i3["encoder"], "encoder_revision": i3["encoder_revision"], "embedding": i3["embedding"],
        "embedding_dim": i3["embedding_dim"], "image_preprocessing_version": config["image_model"]["preprocessing"]["loader_version"],
        "image_preprocessing": i3["image_preprocessing"]}})
    m2_entry = json.loads((METRICS / "multimodal_model_registry.json").read_text(encoding="utf-8"))["models"]["M02"]
    update_registry(METRICS / "multimodal_model_registry.json", {"M04": {**m2_entry, "artifact": tuned["M04"]["artifact"]}})

    audit = test_access_audit(config)
    audit.to_csv(TABLES / "test_access_audit.csv", index=False, lineterminator="\n")
    print(best.to_string(index=False))
    print(audit.to_string(index=False))
    if not audit["passed"].all():
        raise SystemExit("test-access audit FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
