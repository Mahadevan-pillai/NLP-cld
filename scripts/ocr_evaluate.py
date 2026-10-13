"""Task 14 classification under OCR text (called by `python scripts/ocr_experiment.py evaluate`).

Frozen Task 13 models only; no training, tuning or threshold change. Conditions (fixed in the test plan):
    REFERENCE                 dataset caption, model-native preprocessing  -> Task 13 TEST predictions (reused)
    OCR_NATIVE                OCR text through each model's own frozen preprocessing (deployable pipeline)
    OCR_RAW_ONLY / OCR_BASIC_ONLY   hinglish-trained final models (B07, T05, M04) fed OCR text WITHOUT the
                              hinglish step (mismatch ablation: does normalization recover OCR loss?)
    OCR_NATIVE_WITH_FALLBACK  M04 + fallback policy (image-only I03 when OCR fails / finds no text / low conf.)
Outputs: per-condition predictions (git-ignored) and results/metrics/ocr/robustness_predictions_index.json.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models import transformer_text as tt  # noqa: E402
from src.ocr.pipeline import load_ocr_config, preprocess_ocr_text, route  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

PRED_FINAL = REPO_ROOT / "results" / "predictions" / "final"
TEXTS = REPO_ROOT / "results" / "predictions" / "ocr"
METRICS = REPO_ROOT / "results" / "metrics" / "ocr"
NATIVE_MODELS = ["B01", "B02", "B03", "B07", "T04", "T05", "M04"]
FINAL_HINGLISH = ["B07", "T05", "M04"]


def load_fe():
    spec = importlib.util.spec_from_file_location("final_evaluation", REPO_ROOT / "scripts/final_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    config, ocfg = load_config(), load_ocr_config()
    plan = json.loads((METRICS / "test_plan.json").read_text(encoding="utf-8"))
    fe = load_fe()
    fe.verify_freeze(config)
    regs = fe.registries()
    test = fe.load_all_splits(config)["test"].reset_index(drop=True)
    ocr = pd.read_json(TEXTS / f"test_{plan['preparation_selected']}.jsonl", lines=True).set_index("sample_id").loc[test["sample_id"]]
    ocr_text = [t if s in ("ok", "no_text") else "" for t, s in zip(ocr["ocr_text"].fillna(""), ocr["status"])]
    ocr_df = test.assign(text_raw=ocr_text)
    out = {"sample_id": test["sample_id"].to_numpy(), "true_label": test["target_label"].astype(int).to_numpy()}

    def ref_pred(mid):
        p = pd.read_csv(PRED_FINAL / f"{mid}_test_predictions.csv").set_index("sample_id").loc[test["sample_id"]]
        return p["predicted_label"].to_numpy()

    for mid in NATIVE_MODELS + ["I03"]:
        out[f"{mid}|REFERENCE"] = ref_pred(mid)
    feats = fe.Features(ocr_df, config)                                  # OCR text replaces the caption
    for mid in NATIVE_MODELS:
        out[f"{mid}|OCR_NATIVE"] = fe.predict(mid, feats, regs)[0]
        print(f"  {mid} OCR_NATIVE done", flush=True)
    image = feats.image()                                                # reused for the ablations below
    for mode, cond in (("raw", "OCR_RAW_ONLY"), ("basic", "OCR_BASIC_ONLY")):
        forced = tt.preprocess_all(ocr_df["text_raw"].tolist(), mode, feats.opts)

        class Forced(fe.Features):                                       # every model sees OCR text in `mode`
            def processed(self, _mode):
                return forced
        f2 = Forced(ocr_df, config)
        f2._cache["image"] = image
        b07 = joblib.load(REPO_ROOT / regs["text"]["B07"]["artifact"])
        out[f"B07|{cond}"] = b07[1:].predict(forced)                     # skip B07's own TextPreprocessor
        out[f"T05|{cond}"] = fe.predict("T05", f2, regs)[0]
        out[f"M04|{cond}"] = fe.predict("M04", f2, regs)[0]
        print(f"  {cond} done", flush=True)
    # Pre-declared fallback policy (config/ocr.yaml) applied to M04
    decisions = [route({"status": s, "confidence": (None if pd.isna(c) else float(c))},
                       preprocess_ocr_text(t, feats.opts) if s in ("ok", "no_text") else None, ocfg["fallback"])
                 for s, c, t in zip(ocr["status"], ocr["confidence"], ocr_text)]
    use_image = np.array([d["route"] != "multimodal" for d in decisions])
    out["M04|OCR_NATIVE_WITH_FALLBACK"] = np.where(use_image, out["I03|REFERENCE"], out["M04|OCR_NATIVE"])
    out["fallback_route"] = np.array([d["route"] for d in decisions])
    out["fallback_human_review"] = np.array([d["human_review"] for d in decisions])
    pd.DataFrame(out).to_csv(TEXTS / "test_robustness_predictions.csv", index=False, lineterminator="\n")   # git-ignored
    print(pd.Series(out["fallback_route"]).value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
