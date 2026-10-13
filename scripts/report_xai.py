"""Task 15 report: attribution analysis, shortcuts, OCR + XAI, error taxonomy, moderation simulation, figures.

Run after `python scripts/xai_experiment.py compute` (and `runtime`):
    python scripts/report_xai.py

Everything here READS frozen models and stored outputs; nothing is trained, tuned or changed.
Tracked tables hold IDs, labels, numbers and group names. Tables / figures that show dataset text or
meme images use the git-ignored suffixes `_with_text.csv` / `_with_content.png`.
Statistics are descriptive; p-values (Mann-Whitney U, Fisher exact) are exploratory and Holm-adjusted
within each family; they do not establish causes.
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr  # noqa: E402

from src.data.eda_plots import CLASS_COLORS, INK_2, _finish, plt  # noqa: E402
from src.evaluation import final as fs  # noqa: E402
from src.explainability import errors as er  # noqa: E402
from src.explainability import image as xi  # noqa: E402
from src.explainability import multimodal as xm  # noqa: E402
from src.explainability.moderation import load_policy, recommend  # noqa: E402
from src.models import fusion  # noqa: E402
from src.preprocessing.hinglish import NEGATION_WORDS  # noqa: E402
from src.training.text_baselines import WATERMARK_RE  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "explainability"
FIGURES = REPO_ROOT / "results" / "figures" / "explainability"
METRICS = REPO_ROOT / "results" / "metrics" / "explainability"
PER_SAMPLE = REPO_ROOT / "results" / "predictions" / "explainability"      # git-ignored
GREEN = "#2e9e5b"

# Synthetic probes written for Task 15 (NOT dataset content). Illustrative minimal pairs, not a sample.
PROBES = [
    ("negation", "N1", "ladkiyan kamzor hoti hain", "ladkiyan kamzor nahi hoti hain"),
    ("negation", "N2", "aurat gaadi chala sakti hai", "aurat gaadi nahi chala sakti"),
    ("negation", "N3", "women are good drivers", "women are not good drivers"),
    ("negation", "N4", "she is smart", "she is not smart"),
    ("negation", "N5", "ladki ko ghar tak seemit rakho", "ladki ko ghar tak seemit mat rakho"),
    ("negation", "N6", "ye ladki bahut samajhdar hai", "ye ladki bilkul samajhdar nahi hai"),
    ("gender_swap", "G1", "ladki kitchen mein khana bana rahi hai", "ladka kitchen mein khana bana raha hai"),
    ("gender_swap", "G2", "meri behen doctor hai", "mera bhai doctor hai"),
    ("gender_swap", "G3", "aurat ne exam pass kiya", "aadmi ne exam pass kiya"),
    ("gender_swap", "G4", "she is going to the office", "he is going to the office"),
    ("gender_swap", "G5", "girls are playing cricket", "boys are playing cricket"),
    ("gender_swap", "G6", "biwi ne khana banaya", "pati ne khana banaya"),
    ("spelling", "S1", "ye ladki bahut acchi hai", "ye ladkiii bahuttt acchiii hai"),
    ("spelling", "S2", "tum kya kar rahi ho", "tum kya kr rhi ho"),
]
NEGATION_VARIANTS = ["nahi", "nhi", "nahin", "nai", "ni"]
NEGATION_TEMPLATE = "ladkiyan kamzor {} hoti hain"


def load_xe():
    spec = importlib.util.spec_from_file_location("xai_experiment", REPO_ROOT / "scripts/xai_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def holm_dict(p: dict) -> dict:
    return fs.holm({k: v for k, v in p.items() if v is not None and not np.isnan(v)})


def mwu(a, b):
    a, b = [x for x in a if x is not None and not np.isnan(x)], [x for x in b if x is not None and not np.isnan(x)]
    return float(mannwhitneyu(a, b).pvalue) if len(a) >= 5 and len(b) >= 5 else None


def med(x):
    x = [v for v in x if v is not None and not np.isnan(v)]
    return round(float(np.median(x)), 4) if x else None


def lex_match(token: str, lexicon: set) -> bool:
    return token.lower() in lexicon


# ====================================================================== text
def text_analysis(test, cfg, ig_ref):
    lex = {k: {w.lower() for w in v} for k, v in cfg["lexicons"].items()}
    lex["negation"] = {w.lower() for w in NEGATION_WORDS}
    y = dict(zip(test.sample_id, test.y))
    pred = dict(zip(test.sample_id, test.T05_pred))
    rows = []
    for sid, r in ig_ref.items():
        for t in r["tokens"]:
            rows.append({"sample_id": sid, "token": t["token"], "token_l": t["token"].lower(), "contribution": t["contribution"],
                         "y": y[sid], "pred": pred[sid]})
    w = pd.DataFrame(rows)
    # word-level table (dataset tokens -> git-ignored)
    g = w.groupby("token_l").agg(occurrences=("contribution", "size"), mean_contribution=("contribution", "mean"),
                                 share_positive=("contribution", lambda c: float((c > 0).mean())),
                                 memes=("sample_id", "nunique")).reset_index()
    g[g.occurrences >= 5].sort_values("mean_contribution", ascending=False) \
        .to_csv(TABLES / "t05_word_attribution_with_text.csv", index=False, lineterminator="\n")
    # lexical-cue groups (tracked: group names + numbers only)
    cues = []
    for name in ("female_reference", "male_reference", "negation", "watermark"):
        m = w.token_l.isin(lex[name])
        sub = w[m]
        top_pos = 0
        for sid, r in ig_ref.items():
            pos = [t for t in r["tokens"] if t["contribution"] > 0]
            if pred[sid] == 1 and pos and lex_match(max(pos, key=lambda t: t["contribution"])["token"], lex[name]):
                top_pos += 1
        cues.append({"cue_group": name, "occurrences": int(m.sum()), "memes": int(sub.sample_id.nunique()),
                     "mean_contribution": round(float(sub.contribution.mean()), 4) if len(sub) else None,
                     "mean_contribution_in_non_misogynistic": round(float(sub[sub.y == 0].contribution.mean()), 4)
                     if (sub.y == 0).any() else None,
                     "mean_contribution_in_misogynistic": round(float(sub[sub.y == 1].contribution.mean()), 4)
                     if (sub.y == 1).any() else None,
                     "share_positive": round(float((sub.contribution > 0).mean()), 4) if len(sub) else None,
                     "T05_positive_predictions_where_top_word_in_group": top_pos})
    for c in cues:                                   # how often a cue word is among a meme's 3 strongest words
        memes = [sid for sid, r in ig_ref.items() if any(lex_match(t["token"], lex[c["cue_group"]]) for t in r["tokens"])]
        top3 = sum(any(lex_match(t["token"], lex[c["cue_group"]]) for t in
                       sorted(ig_ref[sid]["tokens"], key=lambda t: -abs(t["contribution"]))[:3]) for sid in memes)
        c["share_of_its_memes_with_cue_in_top3_abs"] = round(top3 / max(1, len(memes)), 4)
    other = w[~w.token_l.isin(set().union(*lex.values()))]
    cues.append({"cue_group": "all other words", "occurrences": len(other), "memes": int(other.sample_id.nunique()),
                 "mean_contribution": round(float(other.contribution.mean()), 4),
                 "mean_contribution_in_non_misogynistic": round(float(other[other.y == 0].contribution.mean()), 4),
                 "mean_contribution_in_misogynistic": round(float(other[other.y == 1].contribution.mean()), 4),
                 "share_positive": round(float((other.contribution > 0).mean()), 4),
                 "T05_positive_predictions_where_top_word_in_group": None})
    cues = pd.DataFrame(cues)
    cues.to_csv(TABLES / "lexical_cue_attribution.csv", index=False, lineterminator="\n")

    # gender-term shortcut: false-positive rate on NON-misogynistic memes with vs without a female-reference word
    has_f = {sid: any(lex_match(t["token"], lex["female_reference"]) for t in r["tokens"]) for sid, r in ig_ref.items()}
    has_m = {sid: any(lex_match(t["token"], lex["male_reference"]) for t in r["tokens"]) for sid, r in ig_ref.items()}
    neg = test[test.y == 0]
    sc, pv = [], {}
    for mid in ("T05", "B07", "M04", "I03"):
        for name, flag in (("female-reference word", has_f), ("male-reference word", has_m)):
            with_ = neg[neg.sample_id.map(flag)]
            without = neg[~neg.sample_id.map(flag)]
            a, b = int((with_[f"{mid}_pred"] == 1).sum()), int((without[f"{mid}_pred"] == 1).sum())
            p = fisher_exact([[a, len(with_) - a], [b, len(without) - b]])[1]
            pv[f"{mid}|{name}"] = p
            sc.append({"model": mid, "caption_contains": name, "non_misogynistic_with": len(with_),
                       "fp_rate_with": round(a / max(1, len(with_)), 4), "non_misogynistic_without": len(without),
                       "fp_rate_without": round(b / max(1, len(without)), 4), "fisher_p": p})
    adj = holm_dict(pv)
    sc = pd.DataFrame(sc)
    sc["fisher_p_holm"] = [float(f"{adj.get(f'{r.model}|{r.caption_contains}', np.nan):.3g}") for r in sc.itertuples()]
    sc["fisher_p"] = [float(f"{v:.3g}") for v in sc.fisher_p]
    sc.to_csv(TABLES / "gender_term_shortcut.csv", index=False, lineterminator="\n")

    # negation / Hinglish groups on TEST (accuracy + attribution of negation words)
    grp = []
    for name, mask in (("contains negation word", test.has_negation), ("no negation word", ~test.has_negation),
                       ("Hinglish spelling variant", test.has_hinglish_variant), ("no Hinglish variant", ~test.has_hinglish_variant)):
        sub = test[mask]
        grp.append({"group": name, "memes": len(sub),
                    **{f"{m}_macro_f1": round(fs.macro_f1_fast(sub.y.to_numpy(), sub[f"{m}_pred"].to_numpy()), 4)
                       for m in ("T05", "B07", "M04")}})
    pd.DataFrame(grp).to_csv(TABLES / "negation_hinglish_groups.csv", index=False, lineterminator="\n")
    return w, cues, sc, has_f


def text_correct_vs_incorrect(test, cfg, ig_ref):
    """T05 outcome groups: how often a female-reference word is the strongest positive word; top-1 share."""
    fr = {w.lower() for w in cfg["lexicons"]["female_reference"]}

    def top_pos(r):
        p = [x for x in r["tokens"] if x["contribution"] > 0]
        return max(p, key=lambda x: x["contribution"])["token"].lower() if p else ""

    def top1_share(r):
        a = sorted((abs(x["contribution"]) for x in r["tokens"]), reverse=True)
        return a[0] / sum(a) if a and sum(a) > 0 else np.nan

    rows, y, p = [], test.y.to_numpy(), test.T05_pred.to_numpy()
    empty = np.array([not ig_ref[s]["tokens"] for s in test.sample_id])
    for name, m in (("TP", (p == 1) & (y == 1)), ("FP", (p == 1) & (y == 0)), ("TN", (p == 0) & (y == 0)), ("FN", (p == 0) & (y == 1))):
        ids = test.sample_id[m & ~empty]
        rows.append({"T05_outcome": name, "memes_non_empty": len(ids),
                     "share_top_positive_word_female_reference": round(float(np.mean([top_pos(ig_ref[s]) in fr for s in ids])), 4),
                     "median_top1_share_of_abs_attribution": round(float(np.nanmedian([top1_share(ig_ref[s]) for s in ids])), 4)})
    rows.append({"T05_outcome": "empty caption (all predicted misogynistic?)", "memes_non_empty": int(empty.sum()),
                 "share_top_positive_word_female_reference": None,
                 "median_top1_share_of_abs_attribution": float((p[empty] == 1).mean()) if empty.any() else None})
    pd.DataFrame(rows).to_csv(TABLES / "text_correct_vs_incorrect.csv", index=False, lineterminator="\n")


def probe_analysis():
    from src.explainability.text import explain_text

    rows = []
    for kind, pid, a, b in PROBES:
        for variant, text in (("A", a), ("B", b)):
            e, l = explain_text(text, "T05"), explain_text(text, "B07")
            top = e["top_positive"][0]["token"] if e["top_positive"] else ""
            rows.append({"probe_type": kind, "probe_id": pid, "variant": variant, "text": text,
                         "T05_label": e["predicted_label"], "T05_logit": e["logit_difference"],
                         "T05_top_positive_word": top, "B07_label": l["predicted_label"], "B07_logit": l["logit_difference"],
                         "T05_tokens": json.dumps([(t["token"], round(t["contribution"], 3)) for t in e["tokens"]],
                                                  ensure_ascii=False)})
    for v in NEGATION_VARIANTS:
        e = explain_text(NEGATION_TEMPLATE.format(v), "T05")
        rows.append({"probe_type": "negation_spelling", "probe_id": f"V-{v}", "variant": v,
                     "text": NEGATION_TEMPLATE.format(v), "T05_label": e["predicted_label"], "T05_logit": e["logit_difference"],
                     "T05_top_positive_word": e["top_positive"][0]["token"] if e["top_positive"] else "",
                     "B07_label": explain_text(NEGATION_TEMPLATE.format(v), "B07")["predicted_label"], "B07_logit": None,
                     "T05_tokens": json.dumps([(t["token"], round(t["contribution"], 3)) for t in e["tokens"]], ensure_ascii=False)})
    p = pd.DataFrame(rows)
    p.to_csv(TABLES / "xai_probes.csv", index=False, lineterminator="\n")         # synthetic text: tracked
    pairs = p[p.variant.isin(["A", "B"])].pivot(index=["probe_type", "probe_id"], columns="variant",
                                                 values=["T05_label", "T05_logit", "B07_label"])
    summary = []
    for kind in ("negation", "gender_swap", "spelling"):
        sub = pairs.loc[kind]
        summary.append({"probe_type": kind, "pairs": len(sub),
                        "T05_label_changes": int((sub[("T05_label", "A")] != sub[("T05_label", "B")]).sum()),
                        "B07_label_changes": int((sub[("B07_label", "A")] != sub[("B07_label", "B")]).sum()),
                        "T05_mean_logit_A_minus_B": round(float((sub[("T05_logit", "A")] - sub[("T05_logit", "B")]).mean()), 4)})
    v = p[p.probe_type == "negation_spelling"]
    summary.append({"probe_type": "negation_spelling (5 spellings, same sentence)", "pairs": len(v),
                    "T05_label_changes": int(v.T05_label.nunique() - 1), "B07_label_changes": int(v.B07_label.nunique() - 1),
                    "T05_mean_logit_A_minus_B": round(float(v.T05_logit.max() - v.T05_logit.min()), 4)})
    pd.DataFrame(summary).to_csv(TABLES / "xai_probe_summary.csv", index=False, lineterminator="\n")
    return p


def faithfulness_summary():
    d = pd.read_csv(TABLES / "text_deletion_faithfulness.csv")
    r = pd.read_csv(TABLES / "text_randomised_head_check.csv")
    rows = []
    for k in (1, 3):
        diff = d[f"top{k}_drop"] - d[f"random{k}_drop"]
        rows.append({"check": f"T05 deletion: top-{k} IG words vs {k} random words", "memes": len(d),
                     "median_top_drop": med(d[f"top{k}_drop"]), "median_random_drop": med(d[f"random{k}_drop"]),
                     "share_top_larger": round(float((diff > 0).mean()), 4),
                     "wilcoxon_like_mwu_p": mwu(d[f"top{k}_drop"].tolist(), d[f"random{k}_drop"].tolist())})
    rows.append({"check": "T05 IG with a re-initialised classifier head: Spearman with real attributions",
                 "memes": len(r), "median_top_drop": med(r.spearman), "median_random_drop": None,
                 "share_top_larger": round(float((r.spearman.abs() < 0.5).mean()), 4), "wilcoxon_like_mwu_p": None})
    return rows


# ====================================================================== image
def image_analysis(test, cfg, config, fe):
    index = jsonl(PER_SAMPLE / "occlusion_index_test.jsonl")
    occ_dir = Path((PER_SAMPLE / "occlusion_cache_dir.txt").read_text(encoding="utf-8").strip())
    regs = fe.registries()
    i03 = joblib.load(REPO_ROOT / regs["image"]["I03"]["artifact"])
    fmodel, stds, _ = fusion.load_fusion(REPO_ROOT / regs["multimodal"]["M04"]["artifact"])
    feats = fe.Features(test, config, log=lambda *_: None)
    pooled = feats.finetuned("T05", regs["text"]["T05"]["artifact"], "hinglish_normalized")[0]
    idx = {sid: j for j, sid in enumerate(test.sample_id)}
    w, st = cfg["image"]["window"], cfg["image"]["stride"]
    pos = xi.window_positions(224, w, st)
    scaler = i03[0]
    rng = np.random.default_rng(0)
    random_w = rng.normal(size=(cfg["image"]["sanity_check_random_classifiers"], scaler.mean_.shape[0]))
    rows, maps = [], {}
    for m in index:
        z = np.load(occ_dir / m["cache_file"])
        full, occ = z["full"], z["occluded"]
        j = idx[m["sample_id"]]
        s_full = float(xi.log_odds(i03, full)[0])
        cmap = xi.occlusion_map(s_full, xi.log_odds(i03, occ), pos, 224, w, st)
        t = np.repeat(pooled[j:j + 1], len(occ), 0)
        m_full = float(xm.fusion_logit_difference(fmodel, stds, pooled[j:j + 1], full)[0])
        mmap = xi.occlusion_map(m_full, xm.fusion_logit_difference(fmodel, stds, t, occ), pos, 224, w, st)
        zf, zo = scaler.transform(full), scaler.transform(occ)
        rand = [spearmanr(cmap.ravel(), xi.occlusion_map(float((zf @ rw)[0]), zo @ rw, pos, 224, w, st).ravel()).statistic
                for rw in random_w]
        pred = int(s_full > 0)
        towards = cmap if pred == 1 else -cmap                       # attribution towards the PREDICTED class
        text_r = xi.region_share(towards, m["text_boxes"])
        row = test.iloc[j]
        maps[m["sample_id"]] = (cmap, mmap)
        rows.append({"sample_id": m["sample_id"], "uniform_random": m["uniform_random"], "true_label": int(row.y),
                     "I03_pred": pred, "I03_consistent_with_task13": pred == int(row.I03_pred),
                     "I03_correct": pred == int(row.y), "M04_pred": int(m_full > 0),
                     "M04_consistent_with_task13": int(m_full > 0) == int(row.M04_pred), "M04_correct": int(row.M04_pred) == int(row.y),
                     "concentration_top10": xi.concentration(cmap), "max_abs_cell": round(float(np.abs(cmap).max()), 4),
                     "has_text_boxes": bool(m["text_boxes"]), "text_area_share": text_r["area_share"],
                     "text_mass_share": text_r["positive_mass_share"], "text_enrichment": text_r["enrichment"],
                     "spearman_vs_random_classifiers": round(float(np.nanmean(rand)), 4),
                     "spearman_I03_vs_M04_map": round(float(spearmanr(cmap.ravel(), mmap.ravel()).statistic), 4)})
    r = pd.DataFrame(rows)
    r.to_csv(TABLES / "image_attribution_per_meme.csv", index=False, lineterminator="\n")   # IDs + numbers only
    u = r[r.uniform_random]
    summ = [{"population": "uniform random TEST sample", "memes": len(u),
             "median_concentration_top10": med(u.concentration_top10),
             "median_text_enrichment": med(u.text_enrichment), "share_text_enrichment_gt1": round(float((u.text_enrichment > 1).mean()), 4),
             "median_spearman_vs_random_classifiers": med(u.spearman_vs_random_classifiers),
             "median_spearman_I03_vs_M04": med(u.spearman_I03_vs_M04_map)}]
    pv = {}
    for label, a, b in (("I03 correct vs incorrect", r[r.I03_correct], r[~r.I03_correct]),):
        for col in ("concentration_top10", "text_enrichment", "max_abs_cell"):
            pv[col] = mwu(a[col].tolist(), b[col].tolist())
    adj = holm_dict(pv)
    cvi = []
    for col in ("concentration_top10", "text_enrichment", "max_abs_cell"):
        cvi.append({"model": "I03", "measure": col, "correct_n": int(r.I03_correct.sum()), "incorrect_n": int((~r.I03_correct).sum()),
                    "median_correct": med(r[r.I03_correct][col]), "median_incorrect": med(r[~r.I03_correct][col]),
                    "mwu_p": None if pv[col] is None else float(f"{pv[col]:.3g}"),
                    "mwu_p_holm": float(f"{adj.get(col, np.nan):.3g}")})
    fp = r[(r.I03_pred == 1) & (r.true_label == 0)]
    tn = r[(r.I03_pred == 0) & (r.true_label == 0)]
    summ.append({"population": "I03 false positives (all, research set)", "memes": len(fp), "median_concentration_top10": med(fp.concentration_top10),
                 "median_text_enrichment": med(fp.text_enrichment), "share_text_enrichment_gt1": round(float((fp.text_enrichment > 1).mean()), 4),
                 "median_spearman_vs_random_classifiers": med(fp.spearman_vs_random_classifiers), "median_spearman_I03_vs_M04": med(fp.spearman_I03_vs_M04_map)})
    summ.append({"population": "I03 true negatives (research set)", "memes": len(tn), "median_concentration_top10": med(tn.concentration_top10),
                 "median_text_enrichment": med(tn.text_enrichment), "share_text_enrichment_gt1": round(float((tn.text_enrichment > 1).mean()), 4),
                 "median_spearman_vs_random_classifiers": med(tn.spearman_vs_random_classifiers), "median_spearman_I03_vs_M04": med(tn.spearman_I03_vs_M04_map)})
    pd.DataFrame(summ).to_csv(TABLES / "image_attribution_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(cvi).to_csv(TABLES / "image_correct_vs_incorrect.csv", index=False, lineterminator="\n")
    return r, maps


def overlays(test, maps, sample, review_path):
    """Git-ignored overlays of the explanation sample set + a contact sheet for qualitative review."""
    t = test.set_index("sample_id")
    tiles = []
    for k, sid in enumerate(sample.sample_id):
        if sid not in maps:
            continue
        cmap = maps[sid][0]
        pred = int(t.loc[sid, "I03_pred"])
        ov = xi.overlay(REPO_ROOT / t.loc[sid, "image_path"], cmap)       # red = towards misogynistic
        ov.thumbnail((296, 296))
        tiles.append((f"E{k} {sample.category.iloc[k][:18]} y={int(t.loc[sid, 'y'])} I03={pred}", ov))
    cols = 5
    sheet = Image.new("RGB", (cols * 300, ((len(tiles) + cols - 1) // cols) * 320), "white")
    for n, (label, im) in enumerate(tiles):
        X, Y = (n % cols) * 300, (n // cols) * 320
        sheet.paste(im, (X + 2, Y + 2))
        ImageDraw.Draw(sheet).text((X + 4, Y + 302), label, fill="black")
    sheet.save(FIGURES / "image_attribution_examples_with_content.png")      # git-ignored (dataset images)
    if review_path.exists():
        return pd.read_csv(review_path)
    return None


# ====================================================================== OCR + XAI
def top_words(tokens, k, sign):
    return {t["token"].lower() for t in sorted([t for t in tokens if sign * t["contribution"] > 0],
                                               key=lambda t: -abs(t["contribution"]))[:k]}


def ocr_xai(test, ig_ref):
    ocr = jsonl(PER_SAMPLE / "t05_ig_ocr_test.jsonl")
    t = test.set_index("sample_id")
    rows = []
    for o in ocr:
        sid = o["sample_id"]
        ref = ig_ref[sid]
        old, new = int(t.loc[sid, "T05_pred"]), int(t.loc[sid, "T05_OCR_NATIVE"])
        so, sn = (1 if old else -1), (1 if new else -1)
        ref_words, ocr_words = {x["token"].lower() for x in ref["tokens"]}, {x["token"].lower() for x in o["tokens"]}
        a, b = top_words(ref["tokens"], 3, so), top_words(o["tokens"], 3, sn)
        new_top = max([x for x in o["tokens"] if sn * x["contribution"] > 0], key=lambda x: abs(x["contribution"]), default=None)
        old_top = max([x for x in ref["tokens"] if so * x["contribution"] > 0], key=lambda x: abs(x["contribution"]), default=None)
        rows.append({"sample_id": sid, "group": o["group"], "true_label": int(t.loc[sid, "y"]), "T05_reference": old, "T05_ocr": new,
                     "logit_reference": ref["logit_difference"], "logit_ocr": o["logit_difference"],
                     "top3_jaccard": round(len(a & b) / len(a | b), 4) if a | b else None,
                     "decisive_ocr_word_absent_from_reference": None if new_top is None else new_top["token"].lower() not in ref_words,
                     "decisive_reference_word_absent_from_ocr": None if old_top is None else old_top["token"].lower() not in ocr_words,
                     "words_reference": len(ref_words), "words_ocr": len(ocr_words), "words_shared": len(ref_words & ocr_words),
                     "decisive_ocr_word": None if new_top is None else new_top["token"],
                     "decisive_reference_word": None if old_top is None else old_top["token"]})
    r = pd.DataFrame(rows)
    r.to_csv(TABLES / "ocr_explanation_changes_with_text.csv", index=False, lineterminator="\n")    # git-ignored (tokens)
    out = []
    for g, sub in r.groupby("group"):
        out.append({"group": g, "memes": len(sub), "label_correct_after_ocr": int((sub.T05_ocr == sub.true_label).sum()),
                    "label_correct_before_ocr": int((sub.T05_reference == sub.true_label).sum()),
                    "median_top3_jaccard": med(sub.top3_jaccard),
                    "share_decisive_ocr_word_not_in_reference": round(float(sub.decisive_ocr_word_absent_from_reference.dropna().mean()), 4),
                    "share_decisive_reference_word_lost_by_ocr": round(float(sub.decisive_reference_word_absent_from_ocr.dropna().mean()), 4),
                    "median_share_of_words_shared": med((sub.words_shared / sub[["words_reference", "words_ocr"]].max(axis=1)).tolist())})
    pv = mwu(r[r.group == "label changed"].top3_jaccard.tolist(), r[r.group != "label changed"].top3_jaccard.tolist())
    out = pd.DataFrame(out)
    out["mwu_p_jaccard_changed_vs_control"] = float(f"{pv:.3g}") if pv is not None else None
    out.to_csv(TABLES / "ocr_explanation_changes.csv", index=False, lineterminator="\n")
    # M04 text attribution: reference vs OCR for memes whose M04 label changes with OCR text
    m = pd.DataFrame(jsonl(PER_SAMPLE / "m04_text_ig_test.jsonl"))
    return r, out, m


# ====================================================================== error taxonomy
def error_table(test, sizes, cfg):
    groups = pd.read_csv(REPO_ROOT / load_config()["splitting"]["assignments_file"])
    tg = groups[groups.split == "test"].group_id.value_counts()
    template = set(groups[(groups.split == "test") & groups.group_id.isin(tg[tg > 1].index)].sample_id)
    high_cer = set(pd.read_csv(REPO_ROOT / "results/tables/ocr/ocr_errors.csv").sample_id)
    manual = pd.concat([pd.read_csv(REPO_ROOT / "results/tables/final/manual_error_review.csv").assign(review="Task 13"),
                        pd.read_csv(TABLES / "manual_error_review_task15.csv").assign(review="Task 15")])
    if manual.sample_id.duplicated().any():
        raise SystemExit("a meme was manually reviewed twice")
    manual = manual.set_index("sample_id")
    abl = pd.read_csv(PER_SAMPLE / "multimodal_ablation_test.csv").set_index("sample_id")
    rows = []
    for r in test.itertuples():
        words = len(r.text_raw.split())
        row = {"sample_id": r.sample_id, "true_label": int(r.y), "text_prediction": int(r.T05_pred),
               "image_prediction": int(r.I03_pred), "multimodal_prediction": int(r.M04_pred),
               "caption_empty": not r.text_raw.strip(), "very_short_text": 1 <= words <= 3,
               "text_heavy": words >= 20 or (isinstance(r.ocr_regions, list) and len(r.ocr_regions) > 6),
               "has_negation": bool(r.has_negation), "has_hinglish_variant": bool(r.has_hinglish_variant),
               "has_devanagari": bool(re.search(r"[ऀ-ॿ]", r.text_raw)), "has_watermark": bool(WATERMARK_RE.search(r.text_raw)),
               "low_resolution": min(sizes[r.sample_id]) < 400, "template_group": r.sample_id in template,
               "ocr_changes_prediction": bool(r.M04_OCR_NATIVE != r.M04_pred or r.T05_OCR_NATIVE != r.T05_pred),
               "ocr_high_cer": r.sample_id in high_cer,
               "ocr_low_confidence": bool(r.ocr_confidence is not None and not np.isnan(r.ocr_confidence) and r.ocr_confidence < 0.30)}
        if not any(er.wrong(row, k) for k in ("text", "image", "multimodal")) and not row["ocr_changes_prediction"]:
            continue
        flags = er.auto_flags(row)
        label = manual.error_category.get(r.sample_id)
        cat = er.primary_category(row, flags, label)
        rows.append({**{k: row[k] for k in ("sample_id", "true_label", "text_prediction", "image_prediction", "multimodal_prediction")},
                     **{f"{k}_error": er.error_type(row, k) for k in ("text", "image", "multimodal")},
                     "ocr_status": r.ocr_status, "ocr_route": r.fallback_route, "ocr_changes_prediction": row["ocr_changes_prediction"],
                     "modality_agreement": abl.loc[r.sample_id, "agreement"], "m04_dependence": abl.loc[r.sample_id, "dependence"],
                     **cat, "manual_label": label, "manual_review": manual.review.get(r.sample_id),
                     "auto_flags": ";".join(f"{c}/{s}" for c, s in flags),
                     "manual_review_notes": manual.review_notes.get(r.sample_id, "")})
    e = pd.DataFrame(rows)
    e.to_csv(TABLES / "advanced_error_analysis.csv", index=False, lineterminator="\n")
    return e


def error_distribution(e, test):
    out = []
    m = e[e.multimodal_error.notna()]
    for (c, s), sub in m.groupby(["error_category", "error_subcategory"]):
        out.append({"table": "primary category of M04 errors (manual label where reviewed, else rule)", "model": "M04",
                    "category": c, "subcategory": s, "errors": len(sub), "percent": round(100 * len(sub) / len(m), 1),
                    "false_positives": int((sub.multimodal_error == "false_positive").sum()),
                    "false_negatives": int((sub.multimodal_error == "false_negative").sum()),
                    "manually_reviewed": int(sub.manual_review.notna().sum())})
    reviewed = m[m.manual_review.notna() & m.category_source.eq("manual review")]
    for (c, s), sub in reviewed.groupby(["error_category", "error_subcategory"]):
        out.append({"table": "manually reviewed M04 errors only (60 reviewed)", "model": "M04", "category": c, "subcategory": s,
                    "errors": len(sub), "percent": round(100 * len(sub) / int(m.manual_review.notna().sum()), 1),
                    "false_positives": int((sub.multimodal_error == "false_positive").sum()),
                    "false_negatives": int((sub.multimodal_error == "false_negative").sum()), "manually_reviewed": len(sub)})
    flags = e.assign(flag=e.auto_flags.str.split(";")).explode("flag")
    for model, col in (("T05", "text_error"), ("I03", "image_error"), ("M04", "multimodal_error")):
        errs = flags[flags[col].notna()]
        n_err = int(e[col].notna().sum())
        for f, sub in errs.groupby("flag"):
            if not f or f.startswith("MULTIMODAL/corrected"):
                continue
            c, s = f.split("/", 1)
            out.append({"table": "automatic flags (multi-label) of each model's errors", "model": model, "category": c, "subcategory": s,
                        "errors": len(sub), "percent": round(100 * len(sub) / n_err, 1),
                        "false_positives": int((sub[col] == "false_positive").sum()),
                        "false_negatives": int((sub[col] == "false_negative").sum()), "manually_reviewed": int(sub.manual_review.notna().sum())})
    for model, col in (("T05", "text_error"), ("I03", "image_error"), ("M04", "multimodal_error")):
        out.append({"table": "totals", "model": model, "category": "ALL", "subcategory": "all errors", "errors": int(e[col].notna().sum()),
                    "percent": 100.0, "false_positives": int((e[col] == "false_positive").sum()),
                    "false_negatives": int((e[col] == "false_negative").sum()), "manually_reviewed": int(e[e[col].notna()].manual_review.notna().sum())})
    d = pd.DataFrame(out)
    d.to_csv(TABLES / "error_distribution.csv", index=False, lineterminator="\n")
    return d


# ====================================================================== moderation simulation
def moderation_simulation(test):
    policy = load_policy()
    scen = {
        "meme mode, dataset captions": [{"ocr_route": "image_only" if not r.text_raw.strip() else "multimodal",
                                         "text_label": r.T05_pred, "image_label": r.I03_pred, "multimodal_label": r.M04_pred}
                                        for r in test.itertuples()],
        "meme mode, our OCR + Task 14 fallback": [{"ocr_route": r.fallback_route, "text_label": r.T05_OCR_NATIVE,
                                                    "image_label": r.I03_pred, "multimodal_label": r.M04_OCR_NATIVE}
                                                   for r in test.itertuples()],
    }
    out = []
    for name, inputs in scen.items():
        recs = [recommend({k: (int(v) if k != "ocr_route" else v) for k, v in x.items()}, "meme", policy) for x in inputs]
        out += summarise(name, recs, test.y.to_numpy())
    recs = [recommend({"text_label": int(r.T05_pred), "second_text_label": int(r.B07_pred),
                       "processed_text_empty": not r.text_hinglish_normalized.strip()}, "text", policy) for r in test.itertuples()]
    out += summarise("text mode (T05 + B07), dataset captions", recs, test.y.to_numpy())
    d = pd.DataFrame(out)
    d.to_csv(TABLES / "moderation_test_simulation.csv", index=False, lineterminator="\n")
    return d


def summarise(name, recs, y):
    df = pd.DataFrame({"rec": [r["recommendation"] or "NO RECOMMENDATION" for r in recs], "rule": [r["rule_id"] for r in recs], "y": y})
    rows = []
    for rec, sub in df.groupby("rec"):
        rows.append({"scenario": name, "recommendation": rec, "memes": len(sub), "share": round(len(sub) / len(df), 4),
                     "truly_misogynistic": int(sub.y.sum()), "share_truly_misogynistic": round(float(sub.y.mean()), 4),
                     "rules": ";".join(f"{k}:{v}" for k, v in sub.rule.value_counts().sort_index().items())})
    return rows


# ====================================================================== figures
def figures(probes, cues, img, e, dist, ocr_sum, abl, review):
    # 1. text attribution example (synthetic probes, tracked)
    pick = probes[(probes.probe_id.isin(["N4", "G4"])) & probes.variant.isin(["A", "B"])]
    fig, axes = plt.subplots(1, len(pick), figsize=(3.2 * len(pick), 3.4), sharex=False)
    for ax, (_, p) in zip(np.atleast_1d(axes), pick.iterrows()):
        toks = json.loads(p.T05_tokens)
        ax.barh(range(len(toks))[::-1], [c for _, c in toks], color=[CLASS_COLORS[1] if c > 0 else CLASS_COLORS[0] for _, c in toks])
        ax.set_yticks(range(len(toks))[::-1])
        ax.set_yticklabels([t for t, _ in toks], fontsize=8)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.set_title(f"{p.probe_id}{p.variant}: T05 label {p.T05_label}\nlog-odds {p.T05_logit:+.2f}", fontsize=8)
        ax.set_xlabel("IG attribution (log-odds)", fontsize=7)
    fig.suptitle("T05 word attributions on SYNTHETIC probe sentences (not dataset content)", fontsize=10, fontweight="bold")
    _finish(fig, list(np.atleast_1d(axes)), FIGURES / "text_attribution_example.png")

    # 2. image attribution regions (uniform random sample)
    u = img[img.uniform_random]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, col, lab in ((axes[0], "text_enrichment", "OCR text regions"),):
        v = u[col].dropna().clip(upper=6)
        ax.hist(v, bins=24, color=CLASS_COLORS[0])
        ax.axvline(1, color=INK_2, linestyle="--", linewidth=1, label="= area share (no enrichment)")
        ax.set_xlabel(f"attribution share / area share ({lab}); > 6 shown at 6", fontsize=7)
        ax.set_ylabel("memes", fontsize=8)
        ax.set_title(f"{lab}: median {v.median():.2f} (n = {len(v)})", fontsize=9)
        ax.legend(fontsize=6)
    axes[1].hist(u.spearman_vs_random_classifiers.dropna(), bins=20, color=GREEN)
    axes[1].set_title(f"I03 map vs random classifiers: median rho {u.spearman_vs_random_classifiers.median():.2f}", fontsize=9)
    axes[1].set_xlabel("Spearman rho (lower = map depends on the trained classifier)", fontsize=7)
    fig.suptitle("Where I03's occlusion attribution falls (uniform random TEST sample)", fontsize=10, fontweight="bold")
    _finish(fig, list(axes), FIGURES / "image_attribution_regions.png")

    # 3. correct vs incorrect
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, col, lab in ((axes[0], "concentration_top10", "share of |attribution| in top-10% cells"),
                         (axes[1], "text_enrichment", "OCR-text-region enrichment"), (axes[2], "max_abs_cell", "largest cell |log-odds change|")):
        data = [img[img.I03_correct][col].dropna().clip(upper=6), img[~img.I03_correct][col].dropna().clip(upper=6)]
        ax.boxplot(data, tick_labels=[f"correct\n(n={len(data[0])})", f"incorrect\n(n={len(data[1])})"], showfliers=False)
        ax.set_title(lab, fontsize=8)
    fig.suptitle("I03 occlusion maps: correct vs incorrect predictions (research set; exploratory)", fontsize=10, fontweight="bold")
    _finish(fig, list(axes), FIGURES / "correct_vs_incorrect_attribution.png")

    # 4. modality disagreement
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    order = ["AGREE", "PARTIAL DISAGREEMENT", "DISAGREE"]
    n = [int((abl.agreement == a).sum()) for a in order]
    acc = [float(abl[abl.agreement == a].M04_correct.mean()) if k else 0 for a, k in zip(order, n)]
    axes[0].bar(range(3), n, color=[GREEN, CLASS_COLORS[0], CLASS_COLORS[1]])
    for i, (k, a) in enumerate(zip(n, acc)):
        axes[0].text(i, k, f"n={k}\nM04 acc {a:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(["AGREE", "PARTIAL\nDISAGREEMENT", "DISAGREE"], fontsize=8)
    axes[0].set_ylabel("TEST memes")
    axes[0].set_title("How many single-modality models disagree with M04", fontsize=9)
    dep = abl.dependence.value_counts()
    axes[1].barh(range(len(dep))[::-1], dep.values, color=CLASS_COLORS[0])
    axes[1].set_yticks(range(len(dep))[::-1])
    axes[1].set_yticklabels(dep.index, fontsize=7)
    axes[1].set_xlabel("TEST memes")
    axes[1].set_title("M04 label when one modality is set to its train mean", fontsize=8)
    axes[0].set_ylim(0, max(n) * 1.25)
    fig.suptitle("Modality agreement and dependence (all 758 TEST memes)", fontsize=10, fontweight="bold")
    _finish(fig, list(axes), FIGURES / "modality_disagreement.png")

    # 5. OCR explanation changes
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    o = ocr_sum.set_index("group")
    labels = ["median top-3\nJaccard", "decisive OCR word\nnot in reference", "decisive reference\nword lost by OCR"]
    cols = ["median_top3_jaccard", "share_decisive_ocr_word_not_in_reference", "share_decisive_reference_word_lost_by_ocr"]
    for k, (g, col) in enumerate(zip(o.index, (CLASS_COLORS[1], INK_2))):
        xs, vals = np.arange(3) + (k - 0.5) * 0.36, [float(o.loc[g, c]) for c in cols]
        ax.bar(xs, vals, width=0.36, color=col, label=f"{g} (n={o.loc[g, 'memes']})")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("T05 explanations: reference caption vs our OCR text (TEST)", fontsize=10, fontweight="bold")
    _finish(fig, [ax], FIGURES / "ocr_explanation_changes.png")

    # 6. error categories (primary, M04)
    p = dist[dist.table.str.startswith("primary")].sort_values("errors")
    fig, ax = plt.subplots(figsize=(7.4, 0.3 * len(p) + 1.4))
    yy = range(len(p))
    ax.barh(yy, p.false_negatives, color=CLASS_COLORS[1], label="false negative")
    ax.barh(yy, p.false_positives, left=p.false_negatives, color=CLASS_COLORS[0], label="false positive")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{c} / {s}" for c, s in zip(p.category, p.subcategory)], fontsize=7)
    ax.set_xlabel(f"M04 TEST errors (n = {int(p.errors.sum())}; manual label for {int(p.manually_reviewed.sum())}, rule otherwise)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Error taxonomy: primary category of M04 errors", fontsize=10, fontweight="bold")
    _finish(fig, [ax], FIGURES / "error_category_distribution.png")


def ui_example():
    """Structured explanation object on SYNTHETIC inputs (not dataset content), for Task 16."""
    from src.explainability.explanation import explain_meme, explain_text_input

    img = Image.new("RGB", (400, 300), (205, 190, 170))
    d = ImageDraw.Draw(img)
    d.rectangle((140, 60, 260, 240), fill=(90, 70, 60))
    d.text((20, 270), "SYNTHETIC TEST IMAGE", fill=(0, 0, 0))
    meme = explain_meme(img, caption="ladkiyan kamzor nahi hoti hain")
    text = explain_text_input("she is going to the office")
    (METRICS / "explanation_object_example.json").write_text(json.dumps({"note": "synthetic inputs (generated image, "
        "invented sentences); real frozen models", "meme_mode": meme, "text_mode": text}, indent=1, ensure_ascii=False),
        encoding="utf-8")


def main() -> int:
    for d in (TABLES, FIGURES, METRICS):
        d.mkdir(parents=True, exist_ok=True)
    xe = load_xe()
    config, cfg = load_config(), xe.xcfg()
    fe, _, test = xe.test_context(config)                     # verify_freeze + TEST via the Task 13 access point
    sizes = {}
    for r in test.itertuples():
        with Image.open(REPO_ROOT / r.image_path) as im:
            sizes[r.sample_id] = im.size
    ig_ref = {r["sample_id"]: r for r in jsonl(PER_SAMPLE / "t05_ig_reference_test.jsonl")}
    w, cues, sc, has_f = text_analysis(test, cfg, ig_ref)
    text_correct_vs_incorrect(test, cfg, ig_ref)
    probes = probe_analysis()
    faith = faithfulness_summary()
    pd.DataFrame(faith).to_csv(TABLES / "explanation_faithfulness_checks.csv", index=False, lineterminator="\n")
    img, maps = image_analysis(test, cfg, config, fe)
    sample = pd.read_csv(TABLES / "explanation_sample_set.csv")
    review = overlays(test, maps, sample, TABLES / "image_heatmap_review.csv")
    abl = pd.read_csv(PER_SAMPLE / "multimodal_ablation_test.csv").merge(test[["sample_id", "y", "M04_pred"]], on="sample_id")
    abl["M04_correct"] = abl.M04_pred == abl.y
    agree = []
    for a, sub in abl.groupby("agreement"):
        agree.append({"modality_agreement": a, "memes": len(sub), "share": round(len(sub) / len(abl), 4),
                      "M04_accuracy": round(float(sub.M04_correct.mean()), 4)})
    for dpd, sub in abl.groupby("dependence"):
        agree.append({"modality_agreement": f"dependence: {dpd}", "memes": len(sub), "share": round(len(sub) / len(abl), 4),
                      "M04_accuracy": round(float(sub.M04_correct.mean()), 4)})
    agree.append({"modality_agreement": "M04 label changes when the TEXT is neutralised", "memes": int(abl.label_changes_without_text.sum()),
                  "share": round(float(abl.label_changes_without_text.mean()), 4), "M04_accuracy": None})
    agree.append({"modality_agreement": "M04 label changes when the IMAGE is neutralised", "memes": int(abl.label_changes_without_image.sum()),
                  "share": round(float(abl.label_changes_without_image.mean()), 4), "M04_accuracy": None})
    pd.DataFrame(agree).to_csv(TABLES / "modality_agreement_summary.csv", index=False, lineterminator="\n")
    r_ocr, ocr_sum, m04_ig = ocr_xai(test, ig_ref)
    # M04 text attribution vs T05 attribution (same words, reference captions, sample set)
    corr = []
    for row in m04_ig[m04_ig.text_source == "reference"].itertuples():
        a = [t["contribution"] for t in ig_ref[row.sample_id]["tokens"]]
        b = [t["contribution"] for t in row.tokens]
        if len(a) >= 4 and len(a) == len(b):
            corr.append(spearmanr(a, b).statistic)
    pd.DataFrame([{"comparison": "word attributions: T05 head vs M04 (through fusion, image fixed)", "memes": len(corr),
                   "median_spearman": med(corr)},
                  {"comparison": "M04 text attribution magnitude: median sum|IG| through M04 / through T05",
                   "memes": int((m04_ig.text_source == "reference").sum()),
                   "median_spearman": med([sum(abs(t["contribution"]) for t in r.tokens) /
                                           max(1e-9, sum(abs(t["contribution"]) for t in ig_ref[r.sample_id]["tokens"]))
                                           for r in m04_ig[m04_ig.text_source == "reference"].itertuples()])}]) \
        .to_csv(TABLES / "m04_vs_t05_text_attribution.csv", index=False, lineterminator="\n")
    e = error_table(test, sizes, cfg)
    dist = error_distribution(e, test)
    moderation_simulation(test)
    figures(probes, cues, img, e, dist, ocr_sum, abl, review)
    ui_example()
    print(cues.to_string(index=False))
    print(sc.to_string(index=False))
    print(pd.DataFrame(faith).to_string(index=False))
    print(pd.read_csv(TABLES / "image_attribution_summary.csv").T.to_string())
    print(pd.read_csv(TABLES / "image_correct_vs_incorrect.csv").to_string(index=False))
    print(pd.DataFrame(agree).to_string(index=False))
    print(ocr_sum.to_string(index=False))
    print(dist[dist.table.str.startswith("primary")].to_string(index=False))
    print(pd.read_csv(TABLES / "moderation_test_simulation.csv").to_string(index=False))
    print(pd.read_csv(TABLES / "xai_probe_summary.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
