"""Task 13 report: final TEST tables, statistics, figures, disagreements, error analysis, registry, audit.

Run after `python scripts/final_evaluation.py evaluate`:
    python scripts/report_final_evaluation.py

Every number is recomputed from the saved TEST prediction files and cross-checked against the evaluation
JSONs. Research-question verdicts follow the rules frozen in results/final/model_freeze_manifest.json.
Nothing here trains, tunes or selects a model. VALIDATION and TEST results are kept in separate tables.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.data.eda_plots import CLASS_COLORS, INK_2, SINGLE, _finish, plt  # noqa: E402
from src.evaluation import final as fs  # noqa: E402
from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.data.eda_plots import INK, SURFACE  # noqa: E402
from src.evaluation.plots import SEQUENTIAL  # noqa: E402
from src.training.text_baselines import error_table_flags, subset_masks  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

FINAL = REPO_ROOT / "results" / "final"
PRED_DIR = REPO_ROOT / "results" / "predictions" / "final"
TABLES = REPO_ROOT / "results" / "tables" / "final"
FIGURES = REPO_ROOT / "results" / "figures" / "final"
MANUAL_REVIEW = TABLES / "manual_error_review.csv"          # written by hand after inspecting TEST errors (qualitative)
KEYS = ("accuracy", "precision", "recall", "macro_f1", "weighted_f1")
GREEN = "#2e9e5b"
MODEL_COLORS = {"B07": INK_2, "T05": CLASS_COLORS[0], "I03": GREEN, "M04": CLASS_COLORS[1]}


def test_confusion_figure(tn, fp, fn, tp, names, title, path):
    """Same design as src/evaluation/plots.confusion_matrix_figure (frozen code, not edited), labelled TEST."""
    from matplotlib.colors import LinearSegmentedColormap

    cm = np.array([[tn, fp], [fn, tp]])
    row_pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(row_pct, cmap=LinearSegmentedColormap.from_list("blue", SEQUENTIAL), vmin=0, vmax=100)
    labels = [names[0].replace("_", "-"), names[1].replace("_", "-")]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)", ha="center", va="center", fontsize=10,
                    color=SURFACE if row_pct[i, j] > 55 else INK)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return _finish(fig, [], path, note=f"FINAL TEST set (n = {cm.sum():,}). % = share of the true class (row).")


def load_fe():
    spec = importlib.util.spec_from_file_location("final_evaluation", REPO_ROOT / "scripts/final_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metric_row(y, p, names, n_label="test_sample_count"):
    m = classification_metrics(y, p, names)
    return {n_label: m["n"], **{k: m[k] for k in KEYS}}


def rq_verdict(diff: float, ci_low: float, significant: bool | None) -> str:
    if diff <= 0:
        return "NOT SUPPORTED"
    if significant:
        return "SUPPORTED"
    return "PARTIALLY SUPPORTED"


def bar_with_ci(ax, labels, values, lows, highs, colors):
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.6)
    ax.errorbar(x, values, yerr=[np.array(values) - np.array(lows), np.array(highs) - np.array(values)], fmt="none",
                ecolor=INK_2, capsize=4, linewidth=1.2)
    for i, v in enumerate(values):
        ax.text(i, highs[i] + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)                     # full scale: small differences are not visually exaggerated


def main() -> int:
    config = load_config()
    names = class_names(config)
    fe = load_fe()
    manifest = fe.verify_freeze(config)                              # models / data / code unchanged since the freeze
    for d in (TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    test = fe.load_all_splits(config)["test"].reset_index(drop=True)  # labels + metadata (same frozen access point)
    ids = test["sample_id"].to_numpy()
    y = test["target_label"].astype(int).to_numpy()
    models = list(manifest["models"])
    preds, scores, recs = {}, {}, {}
    consistency = []
    for mid in models:
        p = pd.read_csv(PRED_DIR / f"{mid}_test_predictions.csv").set_index("sample_id")
        if set(p.index) != set(ids) or len(p) != len(ids):
            raise SystemExit(f"{mid}: prediction file does not cover exactly the test samples")
        p = p.loc[ids]
        if not (p["true_label"].to_numpy() == y).all():
            raise SystemExit(f"{mid}: labels in the prediction file differ from the test split")
        preds[mid], scores[mid] = p["predicted_label"].to_numpy(), p["score"].to_numpy()
        recs[mid] = json.loads((FINAL / "metrics" / f"{mid}.json").read_text(encoding="utf-8"))
        m = classification_metrics(y, preds[mid], names)
        consistency.append({"model": mid, "n_predictions": len(p), "confusion_total": m["tn"] + m["fp"] + m["fn"] + m["tp"],
                            "confusion_consistent": fs.confusion_consistent(m, len(y)),
                            "metrics_json_matches_prediction_file": all(abs(m[k] - recs[mid]["metrics"][k]) < 1e-12
                                                                        for k in m if isinstance(m[k], (int, float))),
                            "fast_macro_f1_matches": abs(fs.macro_f1_fast(y, preds[mid]) - m["macro_f1"]) < 1e-4})
    cons = pd.DataFrame(consistency)
    cons.to_csv(TABLES / "consistency_checks.csv", index=False, lineterminator="\n")
    if not cons.drop(columns=["model", "n_predictions", "confusion_total"]).all().all():
        raise SystemExit("consistency check FAILED")

    plan = manifest["evaluation_plan"]
    idx = fs.bootstrap_indices(len(y), plan["bootstrap"]["resamples"], plan["bootstrap"]["seed"])
    ci = {mid: fs.bootstrap_ci(y, preds[mid], idx) for mid in models}
    desc = {mid: manifest["models"][mid]["description"] for mid in models}
    role = {mid: manifest["models"][mid]["role"] for mid in models}
    pd.DataFrame([{"model": mid, "description": desc[mid], "role": role[mid], **ci[mid],
                   "bootstrap": f"{plan['bootstrap']['resamples']} resamples, seed {plan['bootstrap']['seed']}"} for mid in models]) \
        .to_csv(TABLES / "confidence_intervals.csv", index=False, lineterminator="\n")
    masks = subset_masks(test)
    pops = {"all test": masks["all"], "caption not empty": masks["with_text"],
            "without doublemean watermark": masks["without_watermark"]}

    # ---- 13.20 main results / 13.6 fair modality comparison
    main_ids = [("Traditional Text Baseline", "B07"), ("Pretrained Text Model (FINAL_TEXT_MODEL)", "T05"),
                ("Image-Only Model (FINAL_IMAGE_MODEL)", "I03"), ("Multimodal Model (FINAL_MULTIMODAL_MODEL)", "M04")]
    main = pd.DataFrame([{"system": s, "model": mid, "modality": manifest["models"][mid]["modality"], "description": desc[mid],
                          **metric_row(y, preds[mid], names), "macro_f1_ci95": f"{ci[mid]['ci_low']:.4f}-{ci[mid]['ci_high']:.4f}",
                          "result_type": "FINAL TEST RESULT"} for s, mid in main_ids])
    main.to_csv(TABLES / "main_results.csv", index=False, lineterminator="\n")
    main[["model", "modality", "description", "test_sample_count", *KEYS, "macro_f1_ci95"]] \
        .to_csv(TABLES / "final_modality_comparison.csv", index=False, lineterminator="\n")

    # ---- 13.7 multimodal gain + 13.16 statistics (primary comparisons)
    gains, stats, pvals = [], [], {}
    for name, (a, b) in plan["primary_comparisons"].items():
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        raw = {"macro_f1": lambda p: f1_score(y, p, average="macro"), "accuracy": lambda p: accuracy_score(y, p),
               "precision": lambda p: precision_score(y, p), "recall": lambda p: recall_score(y, p),
               "weighted_f1": lambda p: f1_score(y, p, average="weighted")}
        # differences of UNROUNDED metrics (subtracting the 4-decimal table values can differ by 1e-4)
        gains.append({"comparison": f"{a} - {b}", "question": name,
                      **{f"diff_{k}": round(f(preds[a]) - f(preds[b]), 4) for k, f in raw.items()}})
        mc = fs.mcnemar_exact(y, preds[a], preds[b])
        pvals[name] = mc["p_value"]
        stats.append({"comparison": f"{a} vs {b}", "question": name, "type": "PRIMARY (pre-specified)",
                      **fs.paired_bootstrap_difference(y, preds[a], preds[b], idx), **mc})
    holm = fs.holm(pvals)
    for row in stats:
        row["holm_p"] = holm[row["question"]]
        row["significant_by_frozen_rule"] = bool(row["holm_p"] < 0.05 and (row["ci_low"] > 0 or row["ci_high"] < 0))
    pd.DataFrame(gains).to_csv(TABLES / "multimodal_gain.csv", index=False, lineterminator="\n")

    # ---- 13.8 preprocessing ablation (RQ1)
    variant = {"B01": "RAW", "B02": "BASIC", "B03": "HINGLISH_NORMALIZED", "B04": "RAW", "B05": "BASIC",
               "B06": "HINGLISH_NORMALIZED", "T01": "RAW", "T02": "BASIC", "T03": "HINGLISH_NORMALIZED",
               "T04": "BASIC", "T05": "HINGLISH_NORMALIZED"}
    pre_rows, rq1 = [], []
    for fam, mids in plan["rq1_comparisons"].items():
        for mid in mids:
            pre_rows.append({"family": fam, "preprocessing": variant[mid], "model": mid, "description": desc[mid],
                             "sample_count": len(y), **{k: v for k, v in metric_row(y, preds[mid], names).items() if k != "test_sample_count"},
                             "macro_f1_ci95": f"{ci[mid]['ci_low']:.4f}-{ci[mid]['ci_high']:.4f}"})
        h = next(m for m in mids if variant[m] == "HINGLISH_NORMALIZED")
        b = next(m for m in mids if variant[m] == "BASIC")
        d = fs.paired_bootstrap_difference(y, preds[h], preds[b], idx)
        rq1.append({"family": fam, "contrast": f"{h} (HINGLISH) - {b} (BASIC)", **d})
        stats.append({"comparison": f"{h} vs {b}", "question": "RQ1 (descriptive)", "type": "secondary, no significance claim",
                      **d, **fs.mcnemar_exact(y, preds[h], preds[b]), "holm_p": None, "significant_by_frozen_rule": None})
    pd.DataFrame(pre_rows).to_csv(TABLES / "preprocessing_ablation.csv", index=False, lineterminator="\n")
    pd.DataFrame(rq1).to_csv(TABLES / "rq1_normalization_contrasts.csv", index=False, lineterminator="\n")

    # ---- 13.10 fusion ablation
    fus = []
    for mid in plan["fusion_ablation"]:
        d = fs.paired_bootstrap_difference(y, preds[mid], preds["M04"], idx) if mid != "M04" else {}
        fus.append({"model": mid, "description": desc[mid], **metric_row(y, preds[mid], names),
                    "macro_f1_ci95": f"{ci[mid]['ci_low']:.4f}-{ci[mid]['ci_high']:.4f}",
                    "diff_vs_M04": d.get("difference"), "diff_vs_M04_ci95": f"{d['ci_low']:.4f}-{d['ci_high']:.4f}" if d else ""})
        if mid != "M04":
            stats.append({"comparison": f"{mid} vs M04", "question": "fusion ablation (descriptive)",
                          "type": "secondary, no significance claim", **d, **fs.mcnemar_exact(y, preds[mid], preds["M04"]),
                          "holm_p": None, "significant_by_frozen_rule": None})
    pd.DataFrame(fus).to_csv(TABLES / "fusion_ablation.csv", index=False, lineterminator="\n")
    st = pd.DataFrame(stats)
    st.to_csv(TABLES / "statistical_tests.csv", index=False, lineterminator="\n")

    # ---- 13.9 modality ablation (identical samples; three populations) + within-model reliance
    rel = json.loads((FINAL / "metrics" / "M04_modality_reliance.json").read_text(encoding="utf-8"))
    mod_rows = []
    for pop, mask in pops.items():
        for group, mid in (("text only", "B07"), ("text only", "T05"), ("image only", "I03"), ("text + image", "M04")):
            mod_rows.append({"population": pop, "modality": group, "model": mid, "description": desc[mid],
                             **metric_row(y[mask], preds[mid][mask], names)})
    for k, m in rel.items():
        mod_rows.append({"population": "all test", "modality": f"M04 with {k.replace('_', ' ')}",
                         "model": "M04 (diagnostic)", "description": "within-model reliance diagnostic",
                         "test_sample_count": m["n"], **{kk: m[kk] for kk in KEYS}})
    pd.DataFrame(mod_rows).to_csv(TABLES / "modality_ablation.csv", index=False, lineterminator="\n")

    # ---- 13.13 per-class
    per_class = []
    for mid in ("B07", "T05", "I03", "M04"):
        m = classification_metrics(y, preds[mid], names)
        for c in names.values():
            per_class.append({"model": mid, "class": c, "precision": m[f"precision_{c}"], "recall": m[f"recall_{c}"],
                              "f1": m[f"f1_{c}"], "support": m[f"support_{c}"]})
    pc = pd.DataFrame(per_class)
    pc.to_csv(TABLES / "per_class.csv", index=False, lineterminator="\n")

    # ---- 13.11 descriptive sub-label recall (no sub-label model exists)
    sub = []
    for g in ("Objectification", "Prejudice", "Humiliation"):
        mask = test[g].astype(int).to_numpy() == 1
        sub.append({"group": g, "test_samples": int(mask.sum()),
                    **{f"recall_{mid}": round(float((preds[mid][mask] == 1).mean()), 4) for mid in ("B07", "T05", "I03", "M04")}})
    pd.DataFrame(sub).to_csv(TABLES / "subcategory_recall.csv", index=False, lineterminator="\n")

    # ---- 13.15 disagreements (text = T05, image = I03, multimodal = M04)
    t_ok, i_ok, m_ok = preds["T05"] == y, preds["I03"] == y, preds["M04"] == y
    cats = {"A_text_correct_image_wrong_mm_correct": t_ok & ~i_ok & m_ok,
            "B_image_correct_text_wrong_mm_correct": i_ok & ~t_ok & m_ok,
            "C_text_wrong_image_wrong_mm_correct": ~t_ok & ~i_ok & m_ok,
            "D_text_correct_image_correct_mm_wrong": t_ok & i_ok & ~m_ok,
            "E_text_and_image_predictions_disagree": preds["T05"] != preds["I03"],
            "F_all_three_wrong": ~t_ok & ~i_ok & ~m_ok}
    dis = pd.DataFrame({"sample_id": ids, "true_label": y, "text_prediction_T05": preds["T05"],
                        "image_prediction_I03": preds["I03"], "multimodal_prediction_M04": preds["M04"],
                        "caption_empty": ~masks["with_text"], **cats})
    dis.to_csv(TABLES / "modality_disagreements.csv", index=False, lineterminator="\n")
    ds = []
    for k, v in cats.items():
        row = {"category": k, "samples": int(v.sum()), "share": round(float(v.mean()), 4)}
        if k.startswith("E_"):
            row.update(mm_correct=int((v & m_ok).sum()), mm_follows_text=int((v & (preds["M04"] == preds["T05"])).sum()),
                       mm_follows_image=int((v & (preds["M04"] == preds["I03"])).sum()))
        ds.append(row)
    pd.DataFrame(ds).to_csv(TABLES / "disagreement_summary.csv", index=False, lineterminator="\n")

    # ---- 13.14 error analysis: every M04 error, computed flags + manual review where done
    flags = error_table_flags(test)
    sizes = []
    for p in test["image_path"]:
        with Image.open(REPO_ROOT / p) as im:
            sizes.append(min(im.size))
    err = pd.DataFrame({"sample_id": ids, "true_label": y, "text_prediction": preds["T05"], "image_prediction": preds["I03"],
                        "multimodal_prediction": preds["M04"], **{k: v.to_numpy() for k, v in flags.items()},
                        "low_resolution_min_side_lt_400": np.array(sizes) < 400})
    err = err[err.multimodal_prediction != err.true_label].copy()
    err["disagreement_category"] = [next((k.split("_")[0] for k, v in cats.items() if k[0] in "DF" and v[ids.tolist().index(s)]), "")
                                    for s in err.sample_id]
    if MANUAL_REVIEW.exists():
        rv = pd.read_csv(MANUAL_REVIEW)
        err = err.merge(rv, on="sample_id", how="left")
    else:
        err["error_category"], err["review_notes"] = None, None
    err["error_category"] = err["error_category"].fillna("not manually reviewed")
    err["review_notes"] = err["review_notes"].fillna("")
    first = ["sample_id", "true_label", "text_prediction", "image_prediction", "multimodal_prediction", "error_category", "review_notes"]
    err[first + [c for c in err.columns if c not in first]].to_csv(TABLES / "error_analysis.csv", index=False, lineterminator="\n")
    rates = [{"flag": "ALL", "samples": len(y), "M04_errors": int((~m_ok).sum()), "error_rate": round(float((~m_ok).mean()), 4)}]
    for k, v in flags.items():
        v = v.to_numpy()
        rates.append({"flag": k, "samples": int(v.sum()), "M04_errors": int((v & ~m_ok).sum()),
                      "error_rate": round(float((v & ~m_ok).sum() / v.sum()), 4) if v.sum() else None})
    pd.DataFrame(rates).to_csv(TABLES / "error_category_rates.csv", index=False, lineterminator="\n")

    # ---- 13.21 validation (development) vs test, kept separate; tuning generalisation check on TEST
    val_tab = pd.read_csv(REPO_ROOT / "results/tables/tuning/before_after_tuning.csv")
    val_tab.insert(0, "result_type", "VALIDATION DEVELOPMENT RESULT")
    val_tab.to_csv(TABLES / "validation_before_after_tuning.csv", index=False, lineterminator="\n")
    tg = []
    for fam, base, tuned in (("traditional text", "B03", "B07"), ("image-only", "I01", "I03"), ("multimodal", "M02", "M04")):
        d = fs.paired_bootstrap_difference(y, preds[tuned], preds[base], idx)
        tg.append({"family": fam, "baseline_model": base, "tuned_model": tuned,
                   "validation_baseline_macro_f1": manifest["models"][base]["validation_macro_f1"],
                   "validation_tuned_macro_f1": manifest["models"][tuned]["validation_macro_f1"],
                   "test_baseline_macro_f1": ci[base]["macro_f1"], "test_tuned_macro_f1": ci[tuned]["macro_f1"],
                   "test_difference": d["difference"], "test_difference_ci95": f"{d['ci_low']:.4f}-{d['ci_high']:.4f}",
                   "note": "both configurations were frozen before TEST; validation and test columns are separate measurements"})
    pd.DataFrame(tg).to_csv(TABLES / "tuning_generalisation_on_test.csv", index=False, lineterminator="\n")

    # ---- 13.22 research-question verdicts (frozen rules)
    prim = {r["question"]: r for r in stats if r["type"].startswith("PRIMARY")}
    pos = [r["difference"] > 0 for r in rq1]
    if all(pos) and all(r["ci_low"] > 0 for r in rq1):
        rq1_v = "SUPPORTED"
    elif all(pos):
        rq1_v = "PARTIALLY SUPPORTED"
    elif not any(pos):
        rq1_v = "NOT SUPPORTED"
    else:
        rq1_v = "INCONCLUSIVE" if all(r["ci_low"] <= 0 <= r["ci_high"] for r in rq1) else "INCONCLUSIVE (mixed signs)"
    rq = [{"question": "RQ1: Does Hinglish normalization improve classification?", "verdict": rq1_v,
           "evidence": "; ".join(f"{r['family']}: {r['difference']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" for r in rq1)}]
    for q, key in (("RQ2: Does multimodal modelling outperform text-only (M04 vs T05)?", "RQ2_multimodal_vs_text"),
                   ("RQ3: Does multimodal modelling outperform image-only (M04 vs I03)?", "RQ3_multimodal_vs_image")):
        r = prim[key]
        rq.append({"question": q, "verdict": rq_verdict(r["difference"], r["ci_low"], r["significant_by_frozen_rule"]),
                   "evidence": f"macro-F1 difference {r['difference']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]; "
                               f"McNemar p {r['p_value']:.4g} (Holm {r['holm_p']:.4g}); "
                               f"significant by frozen rule: {r['significant_by_frozen_rule']}"})
    rq.append({"question": "RQ4: multicategory (Task 8.5 narrowed RQ4)", "verdict": "NOT EVALUATED",
               "evidence": "no sub-category model was trained (stage 2 disabled); descriptive sub-label recall only"})
    pd.DataFrame(rq).to_csv(TABLES / "research_question_answers.csv", index=False, lineterminator="\n")

    # ---- 13.25 existing-system comparison template (no literature values filled in)
    pd.DataFrame([{"study": "MIMIC dataset paper (DOI 10.1145/3656169)", "year": "TO_BE_VERIFIED", "dataset": "MIMIC",
                   "task": "binary misogyny (multimodal)", "model": "BiT + MuRIL (as named in the paper abstract)",
                   "metric": "macro-F1", "reported_result": "TO_BE_VERIFIED", "our_comparable_result": ci["M04"]["macro_f1"],
                   "comparison_valid": "NO (different split / protocol unless verified)",
                   "notes": "value not recorded in this repository; verify from the paper before use"}]) \
        .to_csv(TABLES / "existing_system_comparison_template.csv", index=False, lineterminator="\n")

    # ---- 13.12 / 13.23 figures
    for mid in ("B07", "T05", "I03", "M04"):
        m = classification_metrics(y, preds[mid], names)
        test_confusion_figure(m["tn"], m["fp"], m["fn"], m["tp"], names, f"{mid}: {desc[mid]}\nFINAL TEST",
                                FIGURES / f"{mid}_test_confusion_matrix.png")
    labels = ["B07\nTF-IDF text", "T05\nMuRIL text", "I03\nSigLIP image", "M04\nimage + text"]
    colors = [MODEL_COLORS[m] for m in ("B07", "T05", "I03", "M04")]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bar_with_ci(ax, labels, [ci[m]["macro_f1"] for m in ("B07", "T05", "I03", "M04")],
                [ci[m]["ci_low"] for m in ("B07", "T05", "I03", "M04")], [ci[m]["ci_high"] for m in ("B07", "T05", "I03", "M04")], colors)
    ax.set_ylabel("Test macro-F1 (95% bootstrap CI)")
    ax.set_title("Final systems: TEST macro-F1 (n = 758)", fontsize=10, fontweight="bold")
    _finish(fig, [ax], FIGURES / "final_macro_f1.png")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    acc = [classification_metrics(y, preds[m], names)["accuracy"] for m in ("B07", "T05", "I03", "M04")]
    ax.bar(np.arange(4), acc, color=colors, width=0.6)
    for i, v in enumerate(acc):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Final systems: TEST accuracy (n = 758)", fontsize=10, fontweight="bold")
    _finish(fig, [ax], FIGURES / "final_accuracy.png")
    # modality comparison across populations
    md = pd.DataFrame([r for r in mod_rows if not str(r["model"]).startswith("M04 (")])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    popn = list(pops)
    width = 0.2
    for j, (mid, col) in enumerate(zip(("B07", "T05", "I03", "M04"), colors)):
        vals = [md[(md.population == p) & (md.model == mid)]["macro_f1"].iloc[0] for p in popn]
        ax.bar(np.arange(len(popn)) + (j - 1.5) * width, vals, width=width, color=col, label=labels[j].replace("\n", " "))
    ax.set_xticks(np.arange(len(popn)))
    ax.set_xticklabels([f"{p}\n(n = {int(pops[p].sum())})" for p in popn], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Text vs image vs multimodal on identical TEST samples", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    _finish(fig, [ax], FIGURES / "modality_comparison.png")
    # preprocessing ablation
    pre = pd.DataFrame(pre_rows)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    fams = list(plan["rq1_comparisons"])
    for j, (v, col) in enumerate(zip(("RAW", "BASIC", "HINGLISH_NORMALIZED"), (INK_2, CLASS_COLORS[0], CLASS_COLORS[1]))):
        vals = [pre[(pre.family == f) & (pre.preprocessing == v)]["macro_f1"] for f in fams]
        xs = [i + (j - 1) * 0.26 for i, s in enumerate(vals) if len(s)]
        ax.bar(xs, [s.iloc[0] for s in vals if len(s)], width=0.26, color=col, label=v)
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels([f + ("\n(no RAW model was trained)" if f == "fine-tuned MuRIL" else "") for f in fams], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("RQ1 preprocessing ablation (TEST, n = 758)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    _finish(fig, [ax], FIGURES / "preprocessing_ablation.png")
    # per-class F1
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for j, (c, col) in enumerate(zip(names.values(), (CLASS_COLORS[0], CLASS_COLORS[1]))):
        vals = [pc[(pc.model == m) & (pc["class"] == c)]["f1"].iloc[0] for m in ("B07", "T05", "I03", "M04")]
        ax.bar(np.arange(4) + (j - 0.5) * 0.36, vals, width=0.36, color=col, label=c.replace("_", "-"))
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test F1")
    ax.set_title("Per-class F1 on TEST", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    _finish(fig, [ax], FIGURES / "per_class_f1.png")
    # forest plot of paired differences
    forest = [(f"{r['comparison']}  [{r['question'].split('_')[0] if r['type'].startswith('PRIMARY') else r['question'].split(' ')[0]}]",
               r["difference"], r["ci_low"], r["ci_high"], r["type"].startswith("PRIMARY")) for r in stats]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(forest) + 1.2))
    for i, (lab, dv, lo, hi, primary) in enumerate(reversed(forest)):
        ax.plot([lo, hi], [i, i], color=CLASS_COLORS[1] if primary else INK_2, linewidth=2)
        ax.scatter([dv], [i], color=CLASS_COLORS[1] if primary else SINGLE, s=25, zorder=3)
    ax.axvline(0, color=INK_2, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(forest)))
    ax.set_yticklabels([f[0] for f in reversed(forest)], fontsize=7)
    ax.set_xlabel("Paired difference in TEST macro-F1 (95% bootstrap CI)")
    ax.set_title("Paired comparisons on TEST (orange = primary, pre-specified)", fontsize=10, fontweight="bold")
    _finish(fig, [ax], FIGURES / "paired_differences_ci.png")

    # ---- 13.27 final registry
    reg = {"note": "FINAL TEST RESULTS of the models frozen in results/final/model_freeze_manifest.json",
           "freeze_commit_contains_manifest": True, "label_map": manifest["label_map"], "models": {}}
    for role_name, mid in manifest["finals"].items():
        mm = manifest["models"][mid]
        reg["models"][role_name] = {"model_version": mid, "description": mm["description"], "registry_entry": mm["registry_entry"],
                                    "artifact_sha256": mm["artifact_sha256"], "preprocessing": manifest["preprocessing"],
                                    "encoders": manifest["encoders"], "score_type": mm["score_type"],
                                    "validation_macro_f1": mm["validation_macro_f1"],
                                    "test_metrics": {k: recs[mid]["metrics"][k] for k in KEYS},
                                    "test_macro_f1_ci95": [ci[mid]["ci_low"], ci[mid]["ci_high"]], "test_n": len(y)}
    (FINAL / "final_model_registry.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")

    print(main[["system", "model", "macro_f1", "accuracy", "macro_f1_ci95"]].to_string(index=False))
    print(st[st.type.str.startswith("PRIMARY")][["comparison", "difference", "ci_low", "ci_high", "p_value", "holm_p",
                                                "significant_by_frozen_rule"]].to_string(index=False))
    print(pd.DataFrame(rq)[["question", "verdict"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
