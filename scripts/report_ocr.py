"""Task 14 report: OCR quality (CER/WER), robustness tables, normalization recovery, figures, examples, errors.

Run after `python scripts/ocr_experiment.py evaluate`:
    python scripts/report_ocr.py

OCR QUALITY compares OCR text with the dataset caption. The dataset caption is itself EasyOCR output
(dataset card), NOT human ground truth, so CER/WER measure agreement with the dataset's OCR.
Tables committed to git hold IDs, flags and numbers only; text examples go to a git-ignored file.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.data.eda_plots import CLASS_COLORS, INK_2, _finish, plt  # noqa: E402
from src.evaluation import final as fs  # noqa: E402
from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.models.text_baselines import preprocessing_options  # noqa: E402
from src.ocr import metrics as om  # noqa: E402
from src.preprocessing import PreprocessingConfig, preprocess  # noqa: E402
from src.preprocessing.hinglish import load_rules  # noqa: E402
from src.training.text_baselines import subset_masks  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "ocr"
FIGURES = REPO_ROOT / "results" / "figures" / "ocr"
METRICS = REPO_ROOT / "results" / "metrics" / "ocr"
TEXTS = REPO_ROOT / "results" / "predictions" / "ocr"
MANUAL = TABLES / "manual_ocr_review.csv"            # written by hand after inspecting OCR failures (qualitative)
KEYS = ("accuracy", "precision", "recall", "macro_f1", "weighted_f1")
GREEN = "#2e9e5b"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def load_fe():
    spec = importlib.util.spec_from_file_location("final_evaluation", REPO_ROOT / "scripts/final_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def region_flags(regions, width, height) -> dict:
    """Geometry flags from EasyOCR boxes: rotated text, very small text, many regions."""
    if not regions:
        return {"rotated_text": False, "very_small_text": False, "multiple_regions": False}
    angles, heights = [], []
    for r in regions:
        (x0, y0), (x1, y1), _, (x3, y3) = r["bbox"]
        angles.append(abs(np.degrees(np.arctan2(y1 - y0, x1 - x0))))
        heights.append(np.hypot(x3 - x0, y3 - y0))
    return {"rotated_text": bool(max(angles) > 10), "very_small_text": bool(np.median(heights) < 0.025 * height),
            "multiple_regions": len(regions) > 6}


def main() -> int:
    config = load_config()
    names = class_names(config)
    opts = preprocessing_options(config)
    pcfg = PreprocessingConfig(**opts)
    plan = json.loads((METRICS / "test_plan.json").read_text(encoding="utf-8"))
    for d in (TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    fe = load_fe()
    fe.verify_freeze(config)
    test = fe.load_all_splits(config)["test"].reset_index(drop=True)
    ids = test["sample_id"].to_numpy()
    ocr = pd.read_json(TEXTS / f"test_{plan['preparation_selected']}.jsonl", lines=True).set_index("sample_id").loc[ids]
    ocr["ocr_text"] = ocr["ocr_text"].fillna("")

    # ---------------- OCR QUALITY
    q = []
    rules = load_rules()
    for r, (sid, o) in zip(test.itertuples(), ocr.iterrows()):
        with Image.open(REPO_ROOT / r.image_path) as im:
            w, h = im.size
            contrast = float(np.asarray(im.convert("L"), dtype=float).std())
        row = {"sample_id": sid, "true_label": int(r.target_label), "status": o["status"], "confidence": o["confidence"],
               "n_regions": o["n_regions"], "reference_empty": not r.text_raw.strip(), "ocr_empty": not o["ocr_text"].strip(),
               "seconds": o["seconds"], "low_resolution": min(w, h) < 400, "low_contrast": contrast < 40,
               "devanagari_reference": bool(DEVANAGARI.search(r.text_raw)), "devanagari_ocr": bool(DEVANAGARI.search(o["ocr_text"])),
               "hinglish_variant_reference": bool(rules.pattern.search(r.text_basic)), "emoji_or_symbol_reference": bool(EMOJI.search(r.text_raw)),
               **region_flags(o["regions"], w, h)}
        for mode in ("raw", "basic", "hinglish_normalized"):
            ref, hyp = preprocess(r.text_raw, mode, pcfg), preprocess(o["ocr_text"], mode, pcfg)
            row[f"cer_{mode}"], row[f"wer_{mode}"] = om.cer(ref, hyp), om.wer(ref, hyp)
        q.append(row)
    q = pd.DataFrame(q)
    summary = {"samples": len(q), "ocr_success_rate (text found)": round(float((q.status == "ok").mean()), 4),
               "no_text_rate": round(float((q.status == "no_text").mean()), 4),
               "ocr_error_or_invalid_rate": round(float(q.status.isin(["ocr_error", "invalid_image"]).mean()), 4),
               "empty_ocr_rate": round(float(q.ocr_empty.mean()), 4), "reference_empty": int(q.reference_empty.sum()),
               "ocr_text_where_reference_empty": int((q.reference_empty & ~q.ocr_empty).sum()),
               "mean_confidence": round(float(q.confidence.dropna().mean()), 4),
               "median_confidence": round(float(q.confidence.dropna().median()), 4),
               "low_confidence_lt_0.30": int((q.confidence < 0.30).sum()),
               # runtime from fresh OCR calls only (cache hits take ~0.01 s and would dilute the mean)
               "fresh_ocr_calls": int((~ocr.from_cache.astype(bool)).sum()),
               "mean_seconds_per_image (fresh OCR)": round(float(ocr.loc[~ocr.from_cache.astype(bool), "seconds"].mean()), 2),
               "median_seconds_per_image (fresh OCR)": round(float(ocr.loc[~ocr.from_cache.astype(bool), "seconds"].median()), 2),
               "max_seconds_per_image (fresh OCR)": round(float(ocr.loc[~ocr.from_cache.astype(bool), "seconds"].max()), 2)}
    for mode in ("raw", "basic", "hinglish_normalized"):
        pairs = [(preprocess(t, mode, pcfg), preprocess(o, mode, pcfg)) for t, o in zip(test.text_raw, ocr.ocr_text)]
        c = om.corpus_rates(pairs)
        summary[f"corpus_cer_{mode}"], summary[f"corpus_wer_{mode}"] = round(c["cer"], 4), round(c["wer"], 4)
        summary[f"median_cer_{mode}"] = round(float(q[f"cer_{mode}"].median()), 4)
        summary[f"median_wer_{mode}"] = round(float(q[f"wer_{mode}"].median()), 4)
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(TABLES / "ocr_quality_summary.csv", lineterminator="\n")
    q["ocr_confidence_lt_0.30"] = q.confidence < 0.30
    groups = {"all": np.ones(len(q), bool), **{k: q[k].to_numpy() for k in
              ("devanagari_reference", "hinglish_variant_reference", "low_resolution", "low_contrast", "multiple_regions",
               "very_small_text", "rotated_text", "emoji_or_symbol_reference", "ocr_confidence_lt_0.30")}}
    pd.DataFrame([{"group": g, "samples": int(m.sum()), "median_cer_hinglish": round(float(q[m].cer_hinglish_normalized.median()), 4)
                   if m.sum() else None, "mean_cer_hinglish": round(float(q[m].cer_hinglish_normalized.mean()), 4) if m.sum() else None,
                   "empty_ocr_rate": round(float(q[m].ocr_empty.mean()), 4) if m.sum() else None}
                  for g, m in groups.items()]).to_csv(TABLES / "ocr_quality_by_group.csv", index=False, lineterminator="\n")

    # ---------------- OCR error table (all samples with high CER or empty OCR), manual categories merged
    err = q[(q.cer_hinglish_normalized.fillna(0) > 0.5) | (q.ocr_empty & ~q.reference_empty) | q.status.isin(["ocr_error", "invalid_image"])].copy()
    flag_cols = ["low_resolution", "low_contrast", "rotated_text", "very_small_text", "multiple_regions",
                 "devanagari_reference", "hinglish_variant_reference", "emoji_or_symbol_reference", "ocr_empty"]
    err["auto_flags"] = err[flag_cols].apply(lambda r: ";".join(c for c in flag_cols if r[c]), axis=1)
    if MANUAL.exists():
        err = err.merge(pd.read_csv(MANUAL), on="sample_id", how="left")
    err["error_category"] = err.get("error_category", pd.Series(index=err.index, dtype=object)).fillna("not manually reviewed")
    err["review_notes"] = err.get("review_notes", pd.Series(index=err.index, dtype=object)).fillna("")
    cols = ["sample_id", "true_label", "status", "confidence", "cer_hinglish_normalized", "wer_hinglish_normalized",
            "error_category", "auto_flags", "review_notes"]
    err[cols].sort_values("cer_hinglish_normalized", ascending=False).to_csv(TABLES / "ocr_errors.csv", index=False, lineterminator="\n")

    # ---------------- examples: tracked IDs/numbers + git-ignored text version
    q["norm_gain"] = q.cer_raw - q.cer_hinglish_normalized
    picks = []
    for label, frame in (("good OCR", q[q.cer_raw.notna()].nsmallest(3, "cer_raw")),
                         ("moderate OCR", q[q.cer_raw.between(0.2, 0.4)].head(3)),
                         ("poor OCR", q[q.cer_raw.notna()].nlargest(3, "cer_raw")),
                         ("normalization improves CER", q.nlargest(3, "norm_gain")),
                         ("normalization worsens CER", q.nsmallest(3, "norm_gain"))):
        for _, r in frame.iterrows():
            picks.append({"example_type": label, "sample_id": r.sample_id, "cer_raw": r.cer_raw, "wer_raw": r.wer_raw,
                          "cer_hinglish": r.cer_hinglish_normalized, "wer_hinglish": r.wer_hinglish_normalized})
    ex = pd.DataFrame(picks)
    ex.to_csv(TABLES / "ocr_examples.csv", index=False, lineterminator="\n")
    tx = test.set_index("sample_id")
    ex.assign(reference_text=ex.sample_id.map(tx.text_raw), ocr_text=ex.sample_id.map(ocr.ocr_text),
              ocr_text_hinglish=ex.sample_id.map(lambda s: preprocess(ocr.loc[s, "ocr_text"], "hinglish_normalized", pcfg))) \
        .to_csv(TABLES / "ocr_examples_with_text.csv", index=False, lineterminator="\n")          # git-ignored

    # ---------------- MODEL PERFORMANCE under OCR
    pr = pd.read_csv(TEXTS / "test_robustness_predictions.csv").set_index("sample_id").loc[ids]
    y = pr["true_label"].to_numpy()
    idx = fs.bootstrap_indices(len(y), 10_000, 42)
    rows, conds, drops = [], [c for c in pr.columns if "|" in c], {}
    for c in conds:
        mid, cond = c.split("|")
        m = classification_metrics(y, pr[c].to_numpy(), names)
        ref = pr[f"{mid}|REFERENCE"].to_numpy() if f"{mid}|REFERENCE" in pr else None
        d = fs.paired_bootstrap_difference(y, ref, pr[c].to_numpy(), idx) if ref is not None and cond != "REFERENCE" else None
        drops[(mid, cond)] = d
        rows.append({"model": mid, "condition": cond, "sample_count": len(y), **{k: m[k] for k in KEYS},
                     "macro_f1_ci95": "{:.4f}-{:.4f}".format(*[fs.bootstrap_ci(y, pr[c].to_numpy(), idx)[k] for k in ("ci_low", "ci_high")]),
                     "drop_vs_reference": d["difference"] if d else None,
                     "drop_ci95": f"{d['ci_low']:.4f}-{d['ci_high']:.4f}" if d else "",
                     # per-sample stability: macro-F1 can stay flat while individual predictions change
                     "predictions_changed_vs_reference": int((pr[c].to_numpy() != ref).sum()) if d else None})
    rob = pd.DataFrame(rows).astype({"predictions_changed_vs_reference": "Int64"})
    rob.to_csv(TABLES / "ocr_robustness_results.csv", index=False, lineterminator="\n")
    f1 = {(r.model, r.condition): r.macro_f1 for r in rob.itertuples()}

    # robustness by population (Task 13 populations + OCR-agreement bins; bins use the HINGLISH-level CER)
    masks = subset_masks(test)
    cer_h = q["cer_hinglish_normalized"].to_numpy(dtype=float)
    pops = {"all test": masks["all"], "caption not empty": masks["with_text"],
            "without doublemean watermark": masks["without_watermark"], "caption empty": ~masks["with_text"],
            "CER <= 0.25": cer_h <= 0.25, "0.25 < CER <= 0.5": (cer_h > 0.25) & (cer_h <= 0.5), "CER > 0.5": cer_h > 0.5,
            "routed away from multimodal by fallback": (pr["fallback_route"] != "multimodal").to_numpy()}
    sub = []
    for pop, m in pops.items():
        for mid, cond in (("B07", "OCR_NATIVE"), ("T05", "OCR_NATIVE"), ("M04", "OCR_NATIVE"), ("M04", "OCR_NATIVE_WITH_FALLBACK")):
            a, b = pr[f"{mid}|REFERENCE"].to_numpy()[m], pr[f"{mid}|{cond}"].to_numpy()[m]
            fa, fb = fs.macro_f1_fast(y[m], a), fs.macro_f1_fast(y[m], b)
            sub.append({"population": pop, "samples": int(m.sum()), "model": mid, "condition": cond,
                        "macro_f1_reference": round(fa, 4), "macro_f1_ocr": round(fb, 4), "drop": round(fa - fb, 4),
                        "predictions_changed": int((a != b).sum())})
    pd.DataFrame(sub).to_csv(TABLES / "ocr_robustness_by_subset.csv", index=False, lineterminator="\n")

    # normalization recovery (final hinglish models): OCR_RAW_ONLY -> OCR_BASIC_ONLY -> OCR_NATIVE (= hinglish)
    rec = []
    for mid in ("B07", "T05", "M04"):
        ref, raw, basic, hing = (f1[(mid, c)] for c in ("REFERENCE", "OCR_RAW_ONLY", "OCR_BASIC_ONLY", "OCR_NATIVE"))
        d = fs.paired_bootstrap_difference(y, pr[f"{mid}|OCR_NATIVE"].to_numpy(), pr[f"{mid}|OCR_RAW_ONLY"].to_numpy(), idx)
        loss = ref - raw
        rec.append({"model": mid, "reference": ref, "ocr_raw_only": raw, "ocr_basic_only": basic, "ocr_hinglish (native)": hing,
                    "loss_without_normalization": round(loss, 4), "recovered_by_normalization": d["difference"],
                    "recovered_ci95": f"{d['ci_low']:.4f}-{d['ci_high']:.4f}",
                    "share_of_loss_recovered": round(d["difference"] / loss, 3) if loss > 0 else None})
    pd.DataFrame(rec).to_csv(TABLES / "normalization_recovery.csv", index=False, lineterminator="\n")

    # text vs multimodal robustness: difference of degradations (paired bootstrap)
    def degr(mid):
        a, b = pr[f"{mid}|REFERENCE"].to_numpy(), pr[f"{mid}|OCR_NATIVE"].to_numpy()
        return fs._macro_f1_batched(y, a, idx) - fs._macro_f1_batched(y, b, idx), fs.macro_f1_fast(y, a) - fs.macro_f1_fast(y, b)
    (dt_boot, dt), (dm_boot, dm) = degr("T05"), degr("M04")
    diff = dt_boot - dm_boot
    tvm = pd.DataFrame([{"text_model": "T05", "text_degradation": round(dt, 4), "multimodal_model": "M04",
                         "multimodal_degradation": round(dm, 4), "difference_text_minus_multimodal": round(dt - dm, 4),
                         "ci95_low": round(float(np.percentile(diff, 2.5)), 4), "ci95_high": round(float(np.percentile(diff, 97.5)), 4),
                         "share_resamples_multimodal_more_robust": round(float((diff > 0).mean()), 4)}])
    tvm.to_csv(TABLES / "text_vs_multimodal_robustness.csv", index=False, lineterminator="\n")
    pd.Series(pr["fallback_route"]).value_counts().rename_axis("route").reset_index(name="test_samples") \
        .to_csv(TABLES / "fallback_routes_test.csv", index=False, lineterminator="\n")

    # ---------------- FIGURES
    for col, lab, fname in (("cer_hinglish_normalized", "CER", "cer_distribution.png"), ("wer_hinglish_normalized", "WER", "wer_distribution.png")):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        v = q[col].dropna().clip(upper=1.5)
        ax.hist(v, bins=30, color=CLASS_COLORS[0])
        ax.axvline(v.median(), color=INK_2, linestyle="--", linewidth=1, label=f"median {v.median():.3f}")
        ax.set_xlabel(f"{lab} vs dataset caption (both HINGLISH-normalised; values > 1.5 shown at 1.5)")
        ax.set_ylabel("TEST memes")
        ax.set_title(f"OCR {lab} distribution (n = {len(v)} with non-empty caption)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        _finish(fig, [ax], FIGURES / fname)
    models = ["B07", "T05", "M04"]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for j, (cond, col) in enumerate((("REFERENCE", INK_2), ("OCR_NATIVE", CLASS_COLORS[1]), ("OCR_NATIVE_WITH_FALLBACK", GREEN))):
        vals = [f1.get((m, cond)) for m in models]
        xs = [i + (j - 1) * 0.26 for i, v in enumerate(vals) if v is not None]
        vs = [v for v in vals if v is not None]
        ax.bar(xs, vs, width=0.26, color=col, label=cond.replace("_", " ").lower())
        for x, v in zip(xs, vs):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, color=INK_2)
    ax.axhline(f1[("I03", "REFERENCE")], color=GREEN, linestyle=":", linewidth=1.2, label=f"image-only I03 ({f1[('I03', 'REFERENCE')]:.3f})")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["B07 TF-IDF text", "T05 MuRIL text", "M04 image + text"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("TEST macro-F1")
    ax.set_title("Reference caption vs OCR text (frozen models, n = 758)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    _finish(fig, [ax], FIGURES / "reference_vs_ocr_macro_f1.png")
    # change vs reference (OCR - reference) per preprocessing level, with paired-bootstrap 95% CIs
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    cond_order = ["OCR_RAW_ONLY", "OCR_BASIC_ONLY", "OCR_NATIVE"]
    for k, (mid, col) in enumerate(zip(models, (INK_2, CLASS_COLORS[0], CLASS_COLORS[1]))):
        xs = [i + (k - 1) * 0.26 for i in range(3)]
        ds = [drops[(mid, c)] for c in cond_order]
        vals = [-d["difference"] for d in ds]
        bars_err = [[v - (-d["ci_high"]) for v, d in zip(vals, ds)], [(-d["ci_low"]) - v for v, d in zip(vals, ds)]]
        ax.bar(xs, vals, width=0.26, color=col, label=mid, yerr=bars_err, capsize=3, error_kw={"elinewidth": 0.8})
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["OCR raw only", "OCR basic only", "OCR hinglish (native)"], fontsize=8)
    ax.set_ylabel("Macro-F1 change vs reference caption")
    ax.set_title("OCR text at each preprocessing level (HINGLISH-trained models)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    _finish(fig, [ax], FIGURES / "raw_vs_normalized_ocr.png")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    dd = [drops[("T05", "OCR_NATIVE")], drops[("M04", "OCR_NATIVE")]]
    ax.bar([0, 1], [dt, dm], color=[CLASS_COLORS[0], CLASS_COLORS[1]], width=0.5,
           yerr=[[v - d["ci_low"] for v, d in zip((dt, dm), dd)], [d["ci_high"] - v for v, d in zip((dt, dm), dd)]],
           capsize=4, error_kw={"elinewidth": 0.9})
    for i, v in enumerate((dt, dm)):
        ax.text(i + 0.27, v, f"{v:+.4f}", ha="left", va="center", fontsize=9, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T05 text-only", "M04 image + text"], fontsize=9)
    ax.set_ylabel("Macro-F1 drop (reference - OCR)")
    ax.set_title("OCR degradation: text-only vs multimodal (TEST, 95% CI)", fontsize=10, fontweight="bold")
    _finish(fig, [ax], FIGURES / "text_vs_multimodal_degradation.png")
    cats = err["error_category"].value_counts()
    cats = cats[cats.index != "not manually reviewed"]
    if len(cats):
        fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(cats) + 1.3))
        ax.barh(range(len(cats)), cats.values[::-1], color=CLASS_COLORS[0])
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats.index[::-1], fontsize=8)
        ax.set_xlabel("Manually reviewed OCR failures")
        ax.set_title(f"OCR failure categories (manual review of {int(cats.sum())} worst-CER TEST memes)",
                     fontsize=10, fontweight="bold")
        _finish(fig, [ax], FIGURES / "ocr_failure_categories.png")

    print(json.dumps(summary, indent=1))
    print(rob[rob.model.isin(["B07", "T05", "M04", "I03"])][["model", "condition", "macro_f1", "drop_vs_reference", "drop_ci95"]].to_string(index=False))
    print(pd.DataFrame(rec).to_string(index=False))
    print(tvm.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
