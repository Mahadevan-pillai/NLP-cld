"""Task 15: explanations of the FROZEN final models on TEST (nothing is trained, tuned or changed).

Run from the repository root, in order:
    python scripts/xai_experiment.py plan      # freeze the analysis plan (methods, sets, lexicons, moderation)
    python scripts/xai_experiment.py compute   # explanations on TEST (cached; resumable; about 1 h on CPU)
    python scripts/xai_experiment.py sanity    # cascading randomisation check for T05 IG (added after compute)
    python scripts/xai_experiment.py runtime   # time the public explain_* functions (warm models)
    python scripts/report_xai.py               # tables, taxonomy, shortcut / OCR / moderation analysis, figures

The TEST split is read only through the Task 13 access point, after the Task 13 freeze checksums are
re-verified. Per-sample explanations contain dataset-derived text and image overlays: they are written
only to git-ignored folders (results/predictions/explainability/) and to the cache outside the repo.
"""

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import Image  # noqa: E402

from src.explainability import image as xi  # noqa: E402
from src.explainability import multimodal as xm  # noqa: E402
from src.explainability import text as xt  # noqa: E402
from src.explainability.moderation import load_policy, validate_policy  # noqa: E402
from src.models import fusion  # noqa: E402
from src.models import transformer_text as tt  # noqa: E402
from src.models.image_encoder import load_encoder  # noqa: E402
from src.ocr.cache import OCRCache  # noqa: E402  (generic one-JSON-per-key store)
from src.preprocessing import hinglish  # noqa: E402
from src.preprocessing.image import load_image  # noqa: E402
from src.inference.text import resolve  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

XAI_CONFIG = REPO_ROOT / "config" / "explainability.yaml"
METRICS = REPO_ROOT / "results" / "metrics" / "explainability"
TABLES = REPO_ROOT / "results" / "tables" / "explainability"
PER_SAMPLE = REPO_ROOT / "results" / "predictions" / "explainability"      # git-ignored
PLAN = METRICS / "xai_plan.json"
FINAL_PRED = REPO_ROOT / "results" / "predictions" / "final"
OCR_PRED = REPO_ROOT / "results" / "predictions" / "ocr"


def load_fe():
    spec = importlib.util.spec_from_file_location("final_evaluation", REPO_ROOT / "scripts/final_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def xcfg() -> dict:
    return yaml.safe_load(XAI_CONFIG.read_text(encoding="utf-8"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_root() -> Path:
    load_dotenv(REPO_ROOT / ".env", override=False)
    value = os.environ.get(xcfg()["cache"]["dir_env"], "").strip()
    return (Path(value).parent / xcfg()["cache"]["subdir"]) if value else REPO_ROOT / "models" / "xai_cache"


# ------------------------------------------------------------------ shared TEST context
def test_context(config):
    """TEST frame + Task 13 predictions + Task 14 OCR info, all aligned on sample_id."""
    fe = load_fe()
    manifest = fe.verify_freeze(config)                       # Task 13 models / code / config unchanged
    test = fe.load_all_splits(config)["test"].reset_index(drop=True)
    ids = test["sample_id"]
    for mid in ("T05", "I03", "M04", "B07"):
        p = pd.read_csv(FINAL_PRED / f"{mid}_test_predictions.csv").set_index("sample_id").loc[ids]
        test[f"{mid}_pred"], test[f"{mid}_score"] = p["predicted_label"].to_numpy(), p["score"].to_numpy()
    rob = pd.read_csv(OCR_PRED / "test_robustness_predictions.csv").set_index("sample_id").loc[ids]
    for c in ("T05|OCR_NATIVE", "M04|OCR_NATIVE", "B07|OCR_NATIVE", "fallback_route"):
        test[c.replace("|", "_")] = rob[c].to_numpy()
    ocr = pd.read_json(OCR_PRED / "test_none.jsonl", lines=True).set_index("sample_id").loc[ids]
    test["ocr_text"], test["ocr_status"], test["ocr_confidence"] = ocr["ocr_text"].fillna("").to_numpy(), \
        ocr["status"].to_numpy(), ocr["confidence"].to_numpy()
    test["ocr_regions"] = list(ocr["regions"])
    test["y"] = test["target_label"].astype(int)
    rules = hinglish.load_rules()
    test["has_negation"] = [hinglish.count_negations(t) > 0 for t in test["text_raw"]]
    test["has_hinglish_variant"] = [bool(rules.pattern.search(t)) for t in test["text_basic"]]
    return fe, manifest, test


def sample_set(test: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """15.10 representative explanation set: deterministic, a meme is used at most once."""
    y, t, i, m = test["y"], test["T05_pred"], test["I03_pred"], test["M04_pred"]
    rules = {
        "true_positive": (m == 1) & (y == 1), "true_negative": (m == 0) & (y == 0),
        "false_positive": (m == 1) & (y == 0), "false_negative": (m == 0) & (y == 1),
        "text_image_disagreement": t != i, "multimodal_correction": (i != y) & (m == y),
        "multimodal_failure": (m != y) & ((t == y) | (i == y)),
        "ocr_failure": (test["M04_OCR_NATIVE"] != m) | (test["T05_OCR_NATIVE"] != t),
        "hinglish_example": test["has_hinglish_variant"], "negation_example": test["has_negation"],
    }
    rng = np.random.default_rng(cfg["seed"])
    used, rows = set(), []
    for cat in cfg["categories"]:
        pool = sorted(set(test.loc[rules[cat].to_numpy(), "sample_id"]) - used)
        for sid in rng.choice(pool, size=min(cfg["per_category"], len(pool)), replace=False):
            used.add(sid)
            rows.append({"sample_id": sid, "category": cat})
    s = pd.DataFrame(rows).merge(test[["sample_id", "y", "T05_pred", "I03_pred", "M04_pred"]], on="sample_id")
    return s.rename(columns={"y": "true_label"})


def research_image_ids(test: pd.DataFrame, cfg: dict, sample_ids) -> list[str]:
    rng = np.random.default_rng(cfg["seed"])
    ids = sorted(test["sample_id"])
    chosen = set(rng.choice(ids, size=cfg["uniform_random"], replace=False))
    for mid in cfg["add_all_errors_of"]:
        chosen |= set(test.loc[test[f"{mid}_pred"] != test["y"], "sample_id"])
    return sorted(chosen | set(sample_ids))


# ------------------------------------------------------------------ plan
def cmd_plan(args):
    if PLAN.exists():
        raise SystemExit(f"{PLAN} exists: the plan is written once, before any TEST explanation")
    config = load_config()
    cfg = xcfg()
    validate_policy(load_policy())
    fe = load_fe()
    manifest = fe.verify_freeze(config)
    METRICS.mkdir(parents=True, exist_ok=True)
    plan = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "statement": "Written BEFORE any TEST explanation was computed. Explanations, error analysis and the "
                         "moderation evaluation change no model, threshold, preprocessing rule or split.",
            "config_sha256": hashlib.sha256(XAI_CONFIG.read_bytes()).hexdigest(),
            "frozen_artifacts": {k: v for k, v in manifest["checksums"].items() if k.startswith("artifact:")
                                 and k.split(":")[1] in ("T05", "I03", "M04", "B07")},
            "freeze_commit_manifest_created": manifest["created_utc"],
            "models": cfg["models"], "text": cfg["text"], "image": cfg["image"], "multimodal": cfg["multimodal"],
            "explanation_sample_set": cfg["explanation_sample_set"], "lexicons": cfg["lexicons"],
            "error_analysis": cfg["error_analysis"], "moderation": cfg["moderation"],
            "faithfulness_checks": {"text_deletion": "remove top-1 / top-3 words by IG towards the predicted class vs "
                                                     "5 random draws of the same number of words (uniform random set)",
                                    "text_randomised_head": "IG with T05's classifier head re-initialised (seed 0), "
                                                            "50 memes: Spearman correlation with the real attributions",
                                    "image_randomised_classifier": "5 random linear classifiers on the same occluded "
                                                                   "embeddings: Spearman correlation with I03 maps"}}
    PLAN.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"plan written {plan['created_utc']}")
    return 0


# ------------------------------------------------------------------ compute
def word_spans(tokenizer, processed: str, max_length: int):
    """Input ids + [start, end) piece spans of each word (same '##' rule as merge_wordpieces)."""
    ids = tokenizer([processed], truncation=True, max_length=max_length)["input_ids"][0]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    spans = []
    for j, p in enumerate(pieces):
        if p in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if p.startswith("##") and spans:
            spans[-1][1] = j + 1
        else:
            spans.append([j, j + 1])
    return ids, spans


@torch.no_grad()
def logit_diff_ids(model, ids_list) -> np.ndarray:
    width = max(len(x) for x in ids_list)
    ids = torch.tensor([x + [0] * (width - len(x)) for x in ids_list])
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids_list])
    logits = model(input_ids=ids, attention_mask=mask).logits
    return (logits[:, 1] - logits[:, 0]).numpy()


def deletion_test(model, tokenizer, processed, tokens, max_length, rng, k_values=(1, 3), draws=5):
    """Drop in the predicted-class log-odds after deleting the top-k IG words vs k random words."""
    ids, spans = word_spans(tokenizer, processed, max_length)
    if len(spans) != len(tokens) or len(spans) < 4:
        return None
    base = float(logit_diff_ids(model, [ids])[0])
    sign = 1.0 if base > 0 else -1.0
    order = np.argsort([-sign * t["contribution"] for t in tokens])
    out = {"base": base}
    for k in k_values:
        def without(words):
            drop = {j for w in words for j in range(*spans[w])}
            return [x for j, x in enumerate(ids) if j not in drop]
        variants = [without(order[:k])] + [without(rng.choice(len(spans), k, replace=False)) for _ in range(draws)]
        s = sign * logit_diff_ids(model, variants)
        out[f"top{k}_drop"] = float(sign * base - s[0])
        out[f"random{k}_drop"] = float(np.mean(sign * base - s[1:]))
    return out


def cmd_compute(args):
    if not PLAN.exists():
        raise SystemExit("run `plan` first")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if hashlib.sha256(XAI_CONFIG.read_bytes()).hexdigest() != plan["config_sha256"]:
        raise SystemExit("config/explainability.yaml changed after the plan was written; refusing")
    config, cfg = load_config(), xcfg()
    fe, manifest, test = test_context(config)
    regs = fe.registries()
    PER_SAMPLE.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    root = cache_root()
    text_cache = OCRCache(root / "text_ig")
    occ_dir = root / "occlusion"
    occ_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = lambda m: print(f"[{time.perf_counter() - t0:6.0f}s] {m}", flush=True)  # noqa: E731

    # ---- features through the Task 13 code path; reproduce the Task 13 predictions exactly
    feats = fe.Features(test, config, log=log)
    checks = []
    preds = {}
    for mid in ("T05", "I03", "M04", "B07"):
        p, s, _ = fe.predict(mid, feats, regs)
        preds[mid] = p
        checks.append({"check": f"{mid} predictions reproduce Task 13 TEST predictions",
                       "matches": int((p == test[f"{mid}_pred"].to_numpy()).sum()), "of": len(test)})
    t05_entry = regs["text"]["T05"]
    pooled, _ = feats.finetuned("T05", t05_entry["artifact"], "hinglish_normalized")
    image_emb = feats.image()
    processed = feats.processed("hinglish_normalized")

    # ---- M04 ablation + agreement for all 758
    m_entry = regs["multimodal"]["M04"]
    fmodel, stds, _ = fusion.load_fusion(REPO_ROOT / m_entry["artifact"])
    full = xm.fusion_logit_difference(fmodel, stds, pooled, image_emb)
    no_text = xm.fusion_logit_difference(fmodel, stds, np.repeat(stds[0].mean[None], len(test), 0), image_emb)
    no_image = xm.fusion_logit_difference(fmodel, stds, pooled, np.repeat(stds[1].mean[None], len(test), 0))
    mm = pd.DataFrame({"sample_id": test["sample_id"], "m04_logit": full, "m04_logit_text_neutralised": no_text,
                       "m04_logit_image_neutralised": no_image})
    mm["label_changes_without_text"] = (no_text > 0) != (full > 0)
    mm["label_changes_without_image"] = (no_image > 0) != (full > 0)
    mm["agreement"] = [xm.modality_agreement(a, b, c) for a, b, c in zip(preds["T05"], preds["I03"], preds["M04"])]
    mm["dependence"] = [xm.dependence(int(f > 0), int(a > 0), int(b > 0)) for f, a, b in zip(full, no_text, no_image)]
    mm.to_csv(PER_SAMPLE / "multimodal_ablation_test.csv", index=False, lineterminator="\n")
    checks.append({"check": "M04 ablation 'full' label == M04 prediction", "matches": int(((full > 0) == preds["M04"]).sum()),
                   "of": len(test)})
    log("M04 ablation done")

    # ---- B07 exact linear contributions for all 758
    b07 = joblib.load(REPO_ROOT / regs["text"]["B07"]["artifact"])
    with open(PER_SAMPLE / "b07_linear_test.jsonl", "w", encoding="utf-8") as out:
        errs, signs = [], 0
        for sid, text, pred in zip(test["sample_id"], test["text_raw"], preds["B07"]):
            r = xt.linear_attribution(b07, text)
            errs.append(abs(r["decomposition_error"]))
            signs += int((r["decision_function"] > 0) == pred)
            out.write(json.dumps({"sample_id": sid, "features": r["features"], "tokens": r["tokens"],
                                  "decision_function": r["decision_function"]}, ensure_ascii=False) + "\n")
    checks.append({"check": "B07 linear decomposition sign == B07 prediction", "matches": signs, "of": len(test)})
    checks.append({"check": "B07 max |decomposition error|", "matches": float(max(errs)), "of": None})
    log("B07 linear done")

    # ---- T05 Integrated Gradients for all 758 reference captions (+ OCR text where the label changes)
    model, tok, _ = tt.load_fine_tuned(resolve(t05_entry["artifact"]))
    steps, max_len = cfg["text"]["steps"], t05_entry["max_length"]
    art = manifest["checksums"]["artifact:T05"]

    def ig(text_processed, forward=None, tag="T05"):
        key = sha("|".join([sha(text_processed), tag, art, f"ig{steps}", "pad", plan["config_sha256"][:12]]))
        hit = text_cache.get(key)
        if hit is None:
            hit = xt.transformer_attribution(model, tok, text_processed, max_len, steps, forward=forward)
            text_cache.put(key, hit)
        return hit

    signs = 0
    with open(PER_SAMPLE / "t05_ig_reference_test.jsonl", "w", encoding="utf-8") as out:
        for n, (sid, pt, pred) in enumerate(zip(test["sample_id"], processed, preds["T05"]), start=1):
            r = ig(pt)
            signs += int((r["logit_difference"] > 0) == pred)
            out.write(json.dumps({"sample_id": sid, **r}, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                log(f"T05 IG reference {n}/{len(test)}")
    checks.append({"check": "T05 IG forward sign == T05 prediction (reference captions)", "matches": signs, "of": len(test)})

    ocr_processed = tt.preprocess_all(test["ocr_text"].tolist(), "hinglish_normalized", feats.opts)
    changed = test["T05_OCR_NATIVE"].to_numpy() != preds["T05"]
    rng = np.random.default_rng(42)
    control = rng.choice(np.flatnonzero(~changed), size=int(changed.sum()), replace=False)
    ocr_rows = np.concatenate([np.flatnonzero(changed), np.sort(control)])
    with open(PER_SAMPLE / "t05_ig_ocr_test.jsonl", "w", encoding="utf-8") as out:
        signs = 0
        for j in ocr_rows:
            r = ig(ocr_processed[j])
            signs += int((r["logit_difference"] > 0) == test["T05_OCR_NATIVE"].iloc[j])
            out.write(json.dumps({"sample_id": test["sample_id"].iloc[j], "group": "label changed" if changed[j]
                                  else "control (label unchanged)", **r}, ensure_ascii=False) + "\n")
    checks.append({"check": "T05 IG forward sign == T05 OCR_NATIVE prediction (OCR text)", "matches": signs,
                   "of": len(ocr_rows)})
    log(f"T05 IG OCR done ({int(changed.sum())} changed + controls)")

    # ---- faithfulness: deletion test (uniform random set) + randomised-head sanity check (50 memes)
    uniform = set(np.random.default_rng(cfg["image"]["research_set"]["seed"]).choice(
        sorted(test["sample_id"]), size=cfg["image"]["research_set"]["uniform_random"], replace=False))
    ref_ig = {json.loads(line)["sample_id"]: json.loads(line) for line in
              open(PER_SAMPLE / "t05_ig_reference_test.jsonl", encoding="utf-8")}
    rng = np.random.default_rng(42)
    rows = []
    for sid, pt in zip(test["sample_id"], processed):
        if sid in uniform:
            d = deletion_test(model, tok, pt, ref_ig[sid]["tokens"], max_len, rng)
            if d:
                rows.append({"sample_id": sid, **d})
    pd.DataFrame(rows).to_csv(TABLES / "text_deletion_faithfulness.csv", index=False, lineterminator="\n")
    random_head = copy.deepcopy(model)
    torch.manual_seed(0)
    random_head.classifier.reset_parameters()
    corr = []
    from scipy.stats import spearmanr
    for sid, pt in zip(test["sample_id"], processed):
        if len(corr) >= 50:
            break
        if sid in uniform and len(ref_ig[sid]["tokens"]) >= 4:
            r = xt.transformer_attribution(random_head, tok, pt, max_len, steps)
            a = [t["contribution"] for t in ref_ig[sid]["tokens"]]
            b = [t["contribution"] for t in r["tokens"]]
            corr.append({"sample_id": sid, "spearman": float(spearmanr(a, b).statistic)})
    pd.DataFrame(corr).to_csv(TABLES / "text_randomised_head_check.csv", index=False, lineterminator="\n")
    log(f"text faithfulness checks done ({len(rows)} deletion, {len(corr)} randomised-head)")

    # ---- sample set: M04 text IG through the fusion (image fixed), reference + OCR text where M04 changes
    s = sample_set(test, cfg["explanation_sample_set"])
    s.to_csv(TABLES / "explanation_sample_set.csv", index=False, lineterminator="\n")
    idx = {sid: j for j, sid in enumerate(test["sample_id"])}
    m04_changed = np.flatnonzero(test["M04_OCR_NATIVE"].to_numpy() != preds["M04"])
    with open(PER_SAMPLE / "m04_text_ig_test.jsonl", "w", encoding="utf-8") as out:
        for j in sorted(set(idx[x] for x in s["sample_id"]) | set(m04_changed)):
            fwd = xm.fusion_text_forward(model, fmodel, stds, image_emb[j:j + 1])
            for src_name, txt in (("reference", processed[j]), ("ocr", ocr_processed[j])):
                if src_name == "ocr" and j not in set(m04_changed):
                    continue
                r = ig(txt, forward=fwd, tag=f"M04|{sha(image_emb[j].tobytes().hex())[:16]}")
                out.write(json.dumps({"sample_id": test["sample_id"].iloc[j], "text_source": src_name, **r},
                                     ensure_ascii=False) + "\n")
    log("M04 text IG done")

    # ---- image occlusion: research set (uniform random + all I03/M04 errors + sample set); embeddings cached
    ids = research_image_ids(test, cfg["image"]["research_set"], s["sample_id"])
    i_entry = regs["image"]["I03"]
    processor, encoder = load_encoder(i_entry["encoder"], i_entry["encoder_revision"])
    w, st, fill = cfg["image"]["window"], cfg["image"]["stride"], cfg["image"]["fill"]
    # No automatic face detection: OpenCV 5 no longer ships the Haar cascade detector, and a separate face
    # model would be an extra download. People / faces are assessed only in the manual heatmap review.
    meta = []
    for n, sid in enumerate(ids, start=1):
        r = test.iloc[idx[sid]]
        key = sha("|".join([r["image_sha256"], i_entry["encoder_revision"], f"{w}/{st}/{fill}", "occlusion-v1"]))
        npz = occ_dir / f"{key}.npz"
        if not npz.exists():
            img = load_image(REPO_ROOT / r["image_path"])
            full_e, occ_e, _ = xi.occlusion_embeddings(xi.pixel_values_for(img, processor), encoder, w, st, fill)
            np.savez_compressed(npz, full=full_e, occluded=occ_e)
        with Image.open(REPO_ROOT / r["image_path"]) as im:
            W, H = im.size
        text_boxes = []
        for reg in (r["ocr_regions"] if isinstance(r["ocr_regions"], list) else []):
            xs, ys = [p[0] for p in reg["bbox"]], [p[1] for p in reg["bbox"]]
            text_boxes.append([min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H])
        meta.append({"sample_id": sid, "cache_file": npz.name, "width": W, "height": H,
                     "uniform_random": sid in uniform, "text_boxes": text_boxes})
        if n % 25 == 0:
            log(f"occlusion {n}/{len(ids)}")
    with open(PER_SAMPLE / "occlusion_index_test.jsonl", "w", encoding="utf-8") as out:
        for m_ in meta:
            out.write(json.dumps(m_) + "\n")
    (PER_SAMPLE / "occlusion_cache_dir.txt").write_text(str(occ_dir), encoding="utf-8")
    pd.DataFrame(checks).to_csv(TABLES / "consistency_checks.csv", index=False, lineterminator="\n")
    log(f"done: {len(ids)} images; checks: {checks}")
    return 0


# ------------------------------------------------------------------ extra randomisation checks (added after `compute`)
def cmd_sanity(args):
    """Cascading model-randomisation check for T05 IG + effective dimensionality of T05's pooled features.

    ADDED AFTER THE PLAN: the planned head-only check gave rho ~0.9, which is not informative on its own.
    Re-initialising more of the model shows whether the attributions depend on the trained weights at all.
    """
    from scipy.stats import spearmanr

    config, cfg = load_config(), xcfg()
    fe, _, test = test_context(config)
    regs = fe.registries()
    t05_entry = regs["text"]["T05"]
    model, tok, _ = tt.load_fine_tuned(resolve(t05_entry["artifact"]))
    feats = fe.Features(test, config, log=lambda *_: None)
    processed = feats.processed("hinglish_normalized")
    pooled, _ = feats.finetuned("T05", t05_entry["artifact"], "hinglish_normalized")
    ref = {json.loads(line)["sample_id"]: json.loads(line) for line in
           open(PER_SAMPLE / "t05_ig_reference_test.jsonl", encoding="utf-8")}
    ids = [r["sample_id"] for r in pd.read_csv(TABLES / "text_randomised_head_check.csv").to_dict("records")]
    variants = {}
    torch.manual_seed(0)
    v = copy.deepcopy(model)
    v.classifier.reset_parameters()
    variants["classifier head re-initialised"] = v
    v = copy.deepcopy(v)
    v.bert.encoder.layer[-1].apply(v._init_weights)
    v.bert.pooler.apply(v._init_weights)
    variants["head + last encoder layer + pooler re-initialised"] = v
    torch.manual_seed(0)
    variants["all weights re-initialised (random BERT)"] = type(model)(model.config).eval()
    rows = []
    j = {sid: k for k, sid in enumerate(test["sample_id"])}
    for name, m in variants.items():
        m.eval()
        rho = []
        for sid in ids:
            r = xt.transformer_attribution(m, tok, processed[j[sid]], t05_entry["max_length"], cfg["text"]["steps"])
            a = [t["contribution"] for t in ref[sid]["tokens"]]
            rho.append(float(spearmanr(a, [t["contribution"] for t in r["tokens"]]).statistic))
        rows.append({"randomisation": name, "memes": len(rho), "median_spearman": round(float(np.median(rho)), 4),
                     "median_abs_spearman": round(float(np.median(np.abs(rho))), 4),
                     "share_abs_spearman_below_0.5": round(float(np.mean(np.abs(rho) < 0.5)), 4)})
        print(rows[-1], flush=True)
    centred = pooled - pooled.mean(0)
    s = np.linalg.svd(centred, compute_uv=False) ** 2
    rows.append({"randomisation": "T05 pooled TEST features: variance share of the 1st / first 3 principal components",
                 "memes": len(pooled), "median_spearman": round(float(s[0] / s.sum()), 4),
                 "median_abs_spearman": round(float(s[:3].sum() / s.sum()), 4), "share_abs_spearman_below_0.5": None})
    pd.DataFrame(rows).to_csv(TABLES / "text_randomisation_checks.csv", index=False, lineterminator="\n")
    return 0


# ------------------------------------------------------------------ runtime of the public functions
def cmd_runtime(args):
    from src.explainability.explanation import explain_meme
    from src.explainability.image import explain_image
    from src.explainability.multimodal import explain_multimodal
    from src.explainability.text import explain_text

    config, cfg = load_config(), xcfg()
    _, _, test = test_context(config)
    s = pd.read_csv(TABLES / "explanation_sample_set.csv")
    rows = test.set_index("sample_id").loc[s["sample_id"].iloc[:6]]
    jobs = {
        "text: T05 Integrated Gradients (32 steps)": lambda r: explain_text(r.text_raw, "T05", cfg["text"]["steps"]),
        "text: B07 exact linear": lambda r: explain_text(r.text_raw, "B07"),
        "image: I03 occlusion research (56/28, 50 passes)": lambda r: explain_image(REPO_ROOT / r.image_path, "I03", 56, 28),
        "image: I03 occlusion UI (112/56, 10 passes)": lambda r: explain_image(REPO_ROOT / r.image_path, "I03", 112, 56),
        "multimodal: comparison + ablation": lambda r: explain_multimodal(REPO_ROOT / r.image_path, r.text_raw),
        "multimodal: + attributions through M04": lambda r: explain_multimodal(REPO_ROOT / r.image_path, r.text_raw,
                                                                              attributions=True),
        "full UI object (caption given, fast image map)": lambda r: explain_meme(REPO_ROOT / r.image_path, r.text_raw),
    }
    out = []
    for name, job in jobs.items():
        job(rows.iloc[0])                                          # warm-up: model loading is excluded
        times = []
        for _, r in rows.iloc[1:].iterrows():
            t0 = time.perf_counter()
            job(r)
            times.append(time.perf_counter() - t0)
        out.append({"explanation": name, "runs": len(times), "mean_seconds": round(float(np.mean(times)), 2),
                    "max_seconds": round(float(np.max(times)), 2)})
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(TABLES / "explanation_runtime.csv", index=False, lineterminator="\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "compute", "sanity", "runtime"])
    args = ap.parse_args()
    return {"plan": cmd_plan, "compute": cmd_compute, "sanity": cmd_sanity, "runtime": cmd_runtime}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
