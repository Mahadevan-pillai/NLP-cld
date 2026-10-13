"""Task 14: OCR pipeline evaluation and OCR-robustness experiment (frozen models; nothing is trained).

Run from the repository root, in order:
    python scripts/ocr_experiment.py select     # VALIDATION subset (150): choose the OCR image preparation by a
                                                # fixed rule; write the TEST plan (results/metrics/ocr/test_plan.json)
    python scripts/ocr_experiment.py ocr-test   # OCR all 758 TEST images with the frozen choice (cached)
    python scripts/ocr_experiment.py evaluate   # frozen models on REFERENCE vs OCR text (TEST)
    python scripts/report_ocr.py                # CER/WER, robustness tables, figures, examples, error analysis

The TEST split is read only through the Task 13 access point (scripts/final_evaluation.py), after the Task 13
freeze checksums are re-verified. Per-sample OCR text (dataset-derived) is written only to git-ignored folders.
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.models.text_baselines import preprocessing_options  # noqa: E402
from src.ocr import metrics as om  # noqa: E402
from src.ocr.cache import OCRCache, cached_ocr  # noqa: E402
from src.ocr.engine import OCR_PIPELINE_VERSION, engine_version, extract_text_from_image, get_reader  # noqa: E402
from src.ocr.pipeline import OCR_CONFIG_PATH, load_ocr_config, preprocess_ocr_text, route  # noqa: E402
from src.preprocessing import PreprocessingConfig, preprocess  # noqa: E402
from src.training.text_baselines import load_development_data  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "ocr"
METRICS = REPO_ROOT / "results" / "metrics" / "ocr"
TEXTS = REPO_ROOT / "results" / "predictions" / "ocr"          # git-ignored (dataset-derived text)
PLAN = METRICS / "test_plan.json"


def load_fe():
    spec = importlib.util.spec_from_file_location("final_evaluation", REPO_ROOT / "scripts/final_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ocr_cache() -> OCRCache:
    load_dotenv(REPO_ROOT / ".env", override=False)
    value = os.environ.get(load_ocr_config()["cache"]["dir_env"], "").strip()
    return OCRCache(Path(value).parent / "cache" / "ocr" if value else REPO_ROOT / "models" / "ocr_cache")


def run_ocr_split(df: pd.DataFrame, preparation: str, name: str) -> pd.DataFrame:
    """OCR every image of df (cached); returns per-sample OCR rows and writes them to a git-ignored JSONL."""
    cache, rows = ocr_cache(), []
    TEXTS.mkdir(parents=True, exist_ok=True)
    reader = get_reader()
    start = time.perf_counter()
    with open(TEXTS / f"{name}_{preparation}.jsonl", "w", encoding="utf-8") as out:
        for i, r in enumerate(df.itertuples(), start=1):
            t0 = time.perf_counter()
            res, hit = cached_ocr(REPO_ROOT / r.image_path, r.sample_id, preparation, cache,
                                  lambda p, prep: extract_text_from_image(p, prep, reader=reader))
            row = {"sample_id": r.sample_id, "status": res["status"], "ocr_text": res["text"],
                   "confidence": res["confidence"], "n_regions": len(res["detected_regions"]),
                   "regions": res["detected_regions"], "preparation": preparation, "from_cache": hit,
                   "seconds": round(time.perf_counter() - t0, 2), "engine_version": res["engine_version"]}
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"  {name}/{preparation}: {i}/{len(df)} ({time.perf_counter() - start:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def quality(df: pd.DataFrame, ocr: pd.DataFrame, opts: dict) -> pd.DataFrame:
    """Per-sample CER/WER at three comparison levels (both sides in the SAME Task 6 form)."""
    cfg = PreprocessingConfig(**opts)
    m = df[["sample_id", "text_raw"]].merge(ocr[["sample_id", "ocr_text", "status", "confidence"]], on="sample_id")
    out = []
    for r in m.itertuples():
        row = {"sample_id": r.sample_id, "status": r.status, "confidence": r.confidence,
               "reference_empty": not str(r.text_raw).strip(), "ocr_empty": not str(r.ocr_text).strip()}
        for mode in ("raw", "basic", "hinglish_normalized"):
            ref, hyp = preprocess(r.text_raw, mode, cfg), preprocess(r.ocr_text, mode, cfg)
            row[f"cer_{mode}"], row[f"wer_{mode}"] = om.cer(ref, hyp), om.wer(ref, hyp)
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- select (VALIDATION)
def cmd_select(args):
    config, ocfg = load_config(), load_ocr_config()
    for d in (TABLES, METRICS):
        d.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        raise SystemExit("test plan already written; the OCR preparation is frozen")
    _, val = load_development_data(config)                     # VALIDATION only
    sel = ocfg["selection"]
    per_class = sel["subset_size"] // 2
    subset = pd.concat([g.sample(per_class, random_state=sel["seed"]) for _, g in val.groupby("target_label")])
    subset = subset.sort_values("sample_id").reset_index(drop=True)
    opts = preprocessing_options(config)
    rows, corpus = [], {}
    for prep in ocfg["preparation_candidates"]:
        ocr = run_ocr_split(subset, prep, "validation_subset")
        q = quality(subset, ocr, opts)
        pairs = [(preprocess(t, "hinglish_normalized", PreprocessingConfig(**opts)),
                  preprocess(o, "hinglish_normalized", PreprocessingConfig(**opts)))
                 for t, o in zip(subset["text_raw"], subset.merge(ocr, on="sample_id")["ocr_text"])]
        c = om.corpus_rates(pairs)
        corpus[prep] = c["cer"]
        rows.append({"preparation": prep, "samples": len(subset), "corpus_cer_hinglish": round(c["cer"], 4),
                     "corpus_wer_hinglish": round(c["wer"], 4), "median_cer_hinglish": round(float(q["cer_hinglish_normalized"].median()), 4),
                     "empty_ocr_rate": round(float(q["ocr_empty"].mean()), 4),
                     "mean_confidence": round(float(ocr["confidence"].dropna().mean()), 4),
                     "mean_seconds_per_image": round(float(ocr["seconds"].mean()), 2)})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "preparation_selection_validation.csv", index=False, lineterminator="\n")
    best = min(corpus.values())
    chosen = "none" if corpus["none"] == best else min(corpus, key=corpus.get)      # tie -> 'none'
    text = OCR_CONFIG_PATH.read_text(encoding="utf-8")
    old = "preparation: null"
    if text.count(old) != 1:
        raise SystemExit("config/ocr.yaml preparation already set")
    OCR_CONFIG_PATH.write_text(text.replace(old, f'preparation: "{chosen}"', 1), encoding="utf-8")
    plan = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "statement": "Written BEFORE any TEST image was OCR'd. TEST results do not change OCR settings, fallback or models.",
        "validation_subset": {"size": len(subset), "per_class": per_class, "seed": sel["seed"],
                              "sample_ids_sha256": __import__("hashlib").sha256("\n".join(subset.sample_id).encode()).hexdigest()},
        "preparation_selected": chosen, "selection_rule": sel["rule"], "validation_corpus_cer": corpus,
        "engine": "easyocr", "engine_version": engine_version(), "languages": ocfg["languages"],
        "ocr_pipeline_version": OCR_PIPELINE_VERSION, "fallback_policy": ocfg["fallback"],
        "test_population": "all 758 canonical TEST memes (no sampling, no easy-example filtering)",
        "conditions": {
            "REFERENCE": "dataset caption, model-native preprocessing (= Task 13 predictions)",
            "OCR_NATIVE": "OCR text through the model's own frozen preprocessing (deployable pipeline)",
            "OCR_RAW_ONLY": "OCR text WITHOUT preprocessing fed to hinglish-trained final models (mismatch ablation)",
            "OCR_BASIC_ONLY": "OCR text with BASIC only (hinglish step skipped) fed to hinglish-trained models (ablation)",
            "OCR_NATIVE_WITH_FALLBACK": "M04 with the fallback policy: image-only I03 when OCR fails / is empty / low confidence"},
        "models": {"final": ["T05", "I03", "M04"], "traditional": ["B07"],
                   "mode_matched": {"RAW": "B01", "BASIC": ["B02", "T04"], "HINGLISH_NORMALIZED": ["B03", "T05"]}},
        "metrics": ["accuracy", "precision", "recall", "macro_f1", "weighted_f1"],
        "uncertainty": "paired bootstrap 10,000 resamples, seed 42 (REFERENCE vs OCR on the same samples)",
    }
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"selected preparation: {chosen}; plan written to {PLAN.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------- OCR on TEST
def cmd_ocr_test(args):
    config, ocfg = load_config(), load_ocr_config()
    if not PLAN.exists() or not ocfg["preparation"]:
        raise SystemExit("run `select` first (it fixes the preparation and writes the test plan)")
    fe = load_fe()
    fe.verify_freeze(config)                                    # Task 13 models / code / config unchanged
    test = fe.load_all_splits(config)["test"].reset_index(drop=True)
    ocr = run_ocr_split(test, ocfg["preparation"], "test")
    fresh = ocr.loc[~ocr["from_cache"], "seconds"]                # cache hits take ~0.01 s; exclude them
    print(ocr["status"].value_counts().to_dict(), f"{len(fresh)} fresh OCR calls, mean {fresh.mean():.1f}s/image")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("select", "ocr-test", "evaluate"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.cmd == "evaluate":
        spec = importlib.util.spec_from_file_location("ocr_eval", REPO_ROOT / "scripts/ocr_evaluate.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main()
    {"select": cmd_select, "ocr-test": cmd_ocr_test}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
