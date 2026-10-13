"""Task 13: final, single evaluation of the frozen models on the locked TEST split.

Run from the repository root, IN THIS ORDER:
    python scripts/final_evaluation.py freeze      # 13.1: re-score VALIDATION with this exact code, record checksums
                                                   #       + the evaluation plan -> results/final/model_freeze_manifest.json
    (commit the freeze; nothing below may change models, preprocessing, thresholds or the plan)
    python scripts/final_evaluation.py integrity   # 13.2: TEST split integrity / leakage checks
    python scripts/final_evaluation.py evaluate    # 13.3-13.5: TEST predictions + metrics for every frozen model
    python scripts/report_final_evaluation.py      # tables, statistics, figures, disagreements, audit, registry

`integrity` and `evaluate` refuse to run unless every checksum in the freeze manifest still matches.
The TEST split is opened ONLY by those two commands (load_split("test", allow_test=True)).
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.data.splits import load_split, load_split_assignments  # noqa: E402
from src.evaluation.final import confusion_consistent, sha256_path  # noqa: E402
from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.features.multimodal import finetuned_pooled_features  # noqa: E402
from src.inference.text import resolve  # noqa: E402
from src.models import fusion  # noqa: E402
from src.models import image_encoder as ie  # noqa: E402
from src.models import transformer_text as tt  # noqa: E402
from src.models.text_baselines import preprocessing_options  # noqa: E402
from src.training.text_baselines import load_development_data, subset_masks  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

FINAL = REPO_ROOT / "results" / "final"
MANIFEST = FINAL / "model_freeze_manifest.json"
PRED_DIR = REPO_ROOT / "results" / "predictions" / "final"
TABLES = REPO_ROOT / "results" / "tables" / "final"
METRICS = REPO_ROOT / "results" / "metrics"
HUB = Path.home() / ".cache" / "huggingface" / "hub"

# Every model evaluated on TEST is listed here BEFORE the test set is opened (developer-approved scope).
MODELS = {
    "B01": ("text", "TF-IDF + LR, RAW", "ablation: RQ1 (TF-IDF LR)"),
    "B02": ("text", "TF-IDF + LR, BASIC", "ablation: RQ1 (TF-IDF LR)"),
    "B03": ("text", "TF-IDF + LR, HINGLISH_NORMALIZED", "ablation: RQ1 (TF-IDF LR); untuned traditional baseline"),
    "B04": ("text", "TF-IDF + Linear SVM, RAW", "ablation: RQ1 (TF-IDF SVM)"),
    "B05": ("text", "TF-IDF + Linear SVM, BASIC", "ablation: RQ1 (TF-IDF SVM)"),
    "B06": ("text", "TF-IDF + Linear SVM, HINGLISH_NORMALIZED", "ablation: RQ1 (TF-IDF SVM)"),
    "B07": ("text", "TF-IDF + LR, HINGLISH_NORMALIZED (tuned)", "MAIN: traditional text baseline"),
    "T01": ("text", "frozen MuRIL + LR, RAW", "ablation: RQ1 (frozen MuRIL)"),
    "T02": ("text", "frozen MuRIL + LR, BASIC", "ablation: RQ1 (frozen MuRIL)"),
    "T03": ("text", "frozen MuRIL + LR, HINGLISH_NORMALIZED", "ablation: RQ1 (frozen MuRIL)"),
    "T04": ("text", "fine-tuned MuRIL, BASIC", "ablation: RQ1 (fine-tuned MuRIL)"),
    "T05": ("text", "fine-tuned MuRIL, HINGLISH_NORMALIZED", "FINAL_TEXT_MODEL"),
    "I01": ("image", "frozen SigLIP + LR (C=1)", "ablation: image classifier (untuned)"),
    "I02": ("image", "frozen SigLIP + Linear SVM", "ablation: image classifier"),
    "I03": ("image", "frozen SigLIP + LR (C=0.001, tuned)", "FINAL_IMAGE_MODEL"),
    "M01": ("image+text", "concat(T05 pooled, SigLIP) + LR", "ablation: fusion (early concatenation)"),
    "M02": ("image+text", "fusion MLP, T05 text, baseline config", "ablation: fusion (untuned MLP)"),
    "M03": ("image+text", "fusion MLP, frozen MuRIL text, baseline config", "ablation: fusion (frozen text)"),
    "M04": ("image+text", "fusion MLP, T05 text, tuned config TM-007", "FINAL_MULTIMODAL_MODEL"),
}
FINALS = {"FINAL_TEXT_MODEL": "T05", "FINAL_IMAGE_MODEL": "I03", "FINAL_MULTIMODAL_MODEL": "M04"}

PLAN = {
    "primary_population": "all 758 canonical TEST samples (empty OCR captions included, as in Tasks 8-12)",
    "secondary_populations": ["caption not empty", "without doublemean watermark"],
    "decision_rule": "argmax of the model output (LR probability > 0.5, SVM margin > 0, softmax argmax); no threshold tuning",
    "metrics": ["accuracy", "precision", "recall", "macro_f1", "weighted_f1", "per-class precision/recall/F1",
                "confusion matrix"],
    "bootstrap": {"resamples": 10000, "seed": 42, "method": "percentile, resampling test samples; paired for differences"},
    "primary_comparisons": {"RQ2_multimodal_vs_text": ["M04", "T05"], "RQ3_multimodal_vs_image": ["M04", "I03"]},
    "significance_rule": "only for primary comparisons: Holm-adjusted exact McNemar p < 0.05 AND paired bootstrap 95% CI "
                         "of the macro-F1 difference excludes 0",
    "rq1_comparisons": {"TF-IDF LR": ["B01", "B02", "B03"], "TF-IDF SVM": ["B04", "B05", "B06"],
                        "frozen MuRIL": ["T01", "T02", "T03"], "fine-tuned MuRIL": ["T04", "T05"]},
    "rq1_controlled_contrast": "HINGLISH_NORMALIZED minus BASIC within each family (they differ only by normalization); "
                               "descriptive paired bootstrap CIs, no significance claim",
    "rq_answer_rule": "SUPPORTED = positive difference meeting the significance rule (RQ2/RQ3) or positive with CI excluding 0 "
                      "in all families (RQ1); PARTIALLY SUPPORTED = positive difference without that evidence; "
                      "NOT SUPPORTED = difference <= 0; INCONCLUSIVE = mixed signs across families with CIs including 0",
    "fusion_ablation": ["M01", "M02", "M03", "M04"],
    "modality_ablation": {"text": ["B07", "T05"], "image": ["I03"], "text+image": ["M04"],
                          "within_model_reliance": "M04 with text or image input replaced by its TRAIN mean"},
    "multicategory": "no sub-category model was trained (Task 8.5 stage 2 disabled): descriptive per-sub-label recall only",
    "after_test": "no tuning, retraining, preprocessing, threshold, architecture or model-selection change",
}


# ------------------------------------------------------------------ feature extraction (one code path)
class Features:
    """All model inputs for one split, computed once from raw captions and image files."""

    def __init__(self, df: pd.DataFrame, config: dict, log=print):
        self.df, self.config, self.log = df.reset_index(drop=True), config, log
        self.texts = self.df["text_raw"].tolist()
        self.opts = preprocessing_options(config)
        self._cache = {}

    def processed(self, mode):
        return self._get(("pre", mode), lambda: tt.preprocess_all(self.texts, mode, self.opts))

    def frozen_mean(self, mode):
        def make():
            from transformers import AutoModel, AutoTokenizer
            t = self.config["text_model"]["transformer"]
            tok, enc = self._get("base_tok", lambda: AutoTokenizer.from_pretrained(t["checkpoint"], revision=t["revision"])), \
                self._get("base_enc", lambda: AutoModel.from_pretrained(t["checkpoint"], revision=t["revision"]))
            return tt.mean_pooled_embeddings(self.processed(mode), tok, enc, t["max_length"], t["frozen"]["batch_size"])
        return self._get(("frozen", mode), make)

    def finetuned(self, exp, artifact, mode):
        """(pooled [CLS], logits) of a fine-tuned MuRIL checkpoint; logits = its own head applied to pooled."""
        def make():
            import torch
            model, tok, _ = tt.load_fine_tuned(resolve(artifact))
            pooled = finetuned_pooled_features(self.processed(mode), model, tok, self.config["text_model"]["transformer"]["max_length"])
            with torch.no_grad():
                logits = model.classifier(torch.from_numpy(pooled)).numpy()
            return pooled, logits
        return self._get(("ft", exp), make)

    def image(self):
        def make():
            enc = self.config["image_model"]["encoder"]
            processor, model = ie.load_encoder(enc["checkpoint"], enc["revision"])
            return ie.image_embeddings([REPO_ROOT / p for p in self.df["image_path"]], processor, model, enc["batch_size"])
        return self._get("image", make)

    def _get(self, key, fn):
        if key not in self._cache:
            t0 = time.perf_counter()
            self._cache[key] = fn()
            if isinstance(key, (tuple, str)) and not str(key).startswith("base"):
                self.log(f"    features {key}: {time.perf_counter() - t0:.0f}s")
        return self._cache[key]


def registries():
    return {m: json.loads((METRICS / f"{m}_model_registry.json").read_text(encoding="utf-8"))["models"]
            for m in ("text", "image", "multimodal")}


def entry_for(mid: str, regs: dict) -> dict:
    return regs["text" if mid[0] in "BT" else "image" if mid[0] == "I" else "multimodal"][mid]


def predict(mid: str, feats: Features, regs: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """(prediction, score, score_type) for one frozen model on one split."""
    e = entry_for(mid, regs)
    kind = e["kind"]
    if kind in ("tfidf_logreg", "tfidf_svm"):
        pipe = joblib.load(resolve(e["artifact"]))
        if kind == "tfidf_logreg":
            return pipe.predict(feats.texts), pipe.predict_proba(feats.texts)[:, 1], "uncalibrated LR probability"
        return pipe.predict(feats.texts), pipe.decision_function(feats.texts), "SVM margin (NOT a probability)"
    if kind == "frozen_logreg" and mid[0] == "T":
        clf = joblib.load(resolve(e["artifact"]))
        x = feats.frozen_mean(e["preprocessing_mode"])
        return clf.predict(x), clf.predict_proba(x)[:, 1], "uncalibrated LR probability"
    if kind == "fine_tuned":
        _, logits = feats.finetuned(mid, e["artifact"], e["preprocessing_mode"])
        proba = tt.softmax(logits)
        return proba.argmax(1), proba[:, 1], "uncalibrated softmax probability"
    if kind in ("frozen_logreg", "frozen_svm"):                       # image classifiers
        clf = joblib.load(REPO_ROOT / e["artifact"])
        x = feats.image()
        return clf.predict(x), ie.classifier_scores(clf, x), ("uncalibrated LR probability" if kind == "frozen_logreg"
                                                               else "SVM margin (NOT a probability)")
    reps = feats.config["multimodal_model"]["text_representations"]
    if e["text_representation"] == "finetuned_pooled":
        text = feats.finetuned("T05", reps["finetuned_pooled"]["artifact"], "hinglish_normalized")[0]
    else:
        text = feats.frozen_mean("hinglish_normalized")
    image = feats.image()
    if kind == "concat_logreg":
        clf = joblib.load(REPO_ROOT / e["artifact"])
        x = np.hstack([text, image])
        return clf.predict(x), clf.predict_proba(x)[:, 1], "uncalibrated LR probability"
    model, stds, _ = fusion.load_fusion(REPO_ROOT / e["artifact"])
    proba = fusion.predict_proba(model, stds, text, image)
    return proba.argmax(1), proba[:, 1], "uncalibrated softmax probability"


def stored_validation(mid: str) -> tuple[dict, pd.DataFrame | None]:
    """Stored validation metrics (+ per-sample predictions where they were saved)."""
    if mid in ("B01", "B02", "B03", "B04", "B05", "B06"):
        r = pd.read_csv(REPO_ROOT / "results/tables/text_baselines/baseline_results.csv").set_index("experiment_id").loc[mid]
        return {"macro_f1": float(r.macro_f1), "accuracy": float(r.accuracy)}, None
    folder = "tuning" if mid in ("B07", "I03", "M04") else {"T": "text_models", "I": "image_models", "M": "multimodal"}[mid[0]]
    r = json.loads((METRICS / folder / f"{mid}.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(METRICS / folder / f"{mid}_validation_predictions.csv").set_index("sample_id")
    return {"macro_f1": r["metrics"]["macro_f1"], "accuracy": r["metrics"]["accuracy"]}, preds


# ------------------------------------------------------------------ 13.1 freeze
def artifact_files(mid: str, regs: dict) -> list[Path]:
    e = entry_for(mid, regs)
    return [resolve(e["artifact"]) if e["artifact"].startswith("{MODEL_DIR}") else REPO_ROOT / e["artifact"]]


def frozen_inputs(config) -> dict:
    """Files whose content defines the frozen system (code, configs, rules, split, encoders)."""
    t, i = config["text_model"]["transformer"], config["image_model"]["encoder"]
    files = {
        "config.yaml": REPO_ROOT / "config.yaml",
        "config/categories.yaml": REPO_ROOT / "config/categories.yaml",
        "hinglish_rules": REPO_ROOT / config["text_preprocessing"]["normalization_rules"],
        "split_assignments.csv": REPO_ROOT / config["splitting"]["assignments_file"],
        "split_manifest.json": REPO_ROOT / config["splitting"]["manifest_file"],
        "src/preprocessing": REPO_ROOT / "src/preprocessing",
        "src/models": REPO_ROOT / "src/models",
        "src/features": REPO_ROOT / "src/features",
        "src/evaluation": REPO_ROOT / "src/evaluation",
        "scripts/final_evaluation.py": REPO_ROOT / "scripts/final_evaluation.py",
        "text_model_registry.json": METRICS / "text_model_registry.json",
        "image_model_registry.json": METRICS / "image_model_registry.json",
        "multimodal_model_registry.json": METRICS / "multimodal_model_registry.json",
        "muril_weights": HUB / "models--google--muril-base-cased" / "snapshots" / t["revision"] / "pytorch_model.bin",
        "siglip_weights": HUB / "models--google--siglip-base-patch16-224" / "snapshots" / i["revision"] / "model.safetensors",
    }
    return {k: v for k, v in files.items()}


def code_hash(path: Path) -> str:
    """Folder hash over .py files only (ignores __pycache__)."""
    if path.is_dir():
        h = hashlib.sha256()
        for p in sorted(path.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".yaml"):
                h.update(p.relative_to(path).as_posix().encode())
                h.update(sha256_path(p).encode())
        return h.hexdigest()
    return sha256_path(path)


def current_checksums(config, regs) -> dict:
    out = {k: code_hash(v) for k, v in frozen_inputs(config).items()}
    for mid in MODELS:
        for p in artifact_files(mid, regs):
            out[f"artifact:{mid}"] = sha256_path(p)
    return out


def cmd_freeze(args):
    config = load_config()
    regs = registries()
    names = class_names(config)
    FINAL.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and not args.force:
        raise SystemExit("freeze manifest already exists; refusing to overwrite (the freeze happens once)")
    _, val = load_development_data(config)                      # VALIDATION only: proves this code path pre-unlock
    y = val["target_label"].astype(int).to_numpy()
    feats = Features(val, config)
    models = {}
    for mid, (modality, desc, role) in MODELS.items():
        pred, score, score_type = predict(mid, feats, regs)
        m = classification_metrics(y, pred, names)
        stored, stored_pred = stored_validation(mid)
        same_metric = abs(m["macro_f1"] - stored["macro_f1"]) < 1e-9 and abs(m["accuracy"] - stored["accuracy"]) < 1e-9
        same_pred = None
        if stored_pred is not None:
            sp = stored_pred.loc[val["sample_id"]]
            same_pred = bool((sp["pred"].values == pred).all() and np.abs(sp["score"].values - score).max() < 1e-4)
        ok = same_metric and (same_pred is not False)
        print(f"  {mid}: validation macro-F1 {m['macro_f1']:.4f} (stored {stored['macro_f1']:.4f}) "
              f"metrics match={same_metric} predictions match={same_pred}")
        if not ok:
            raise SystemExit(f"{mid}: the evaluation code does not reproduce the stored validation result; NOT freezing")
        e = entry_for(mid, regs)
        models[mid] = {"modality": modality, "description": desc, "role": role, "registry_entry": e,
                       "artifact_sha256": sha256_path(artifact_files(mid, regs)[0]), "score_type": score_type,
                       "validation_macro_f1": m["macro_f1"], "validation_accuracy": m["accuracy"],
                       "validation_reproduced_by_evaluation_code": True}
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "statement": "Frozen BEFORE any test label was read. After this point TEST results must not influence development.",
        "git_head_at_freeze": head, "finals": FINALS, "label_map": {str(k): v for k, v in names.items()},
        "preprocessing": {"text_version": config["text_preprocessing"]["version"],
                          "image_loader_version": config["image_model"]["preprocessing"]["loader_version"],
                          "image_processor_backend": config["image_model"]["preprocessing"]["processor_backend"]},
        "encoders": {"text": f"{config['text_model']['transformer']['checkpoint']}@{config['text_model']['transformer']['revision']}",
                     "image": f"{config['image_model']['encoder']['checkpoint']}@{config['image_model']['encoder']['revision']}"},
        "tuned_configurations": config["tuning"]["selected"], "random_seed": config["random_seed"],
        "torch_threads_note": "fusion training is bit-exact only for the same CPU thread count; inference is deterministic",
        "evaluation_plan": PLAN, "models": models, "checksums": current_checksums(config, regs),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"freeze manifest written: {MANIFEST.relative_to(REPO_ROOT)} ({len(models)} models)")


def verify_freeze(config) -> dict:
    if not MANIFEST.exists():
        raise SystemExit("no freeze manifest: run `freeze` (and commit it) before touching the test set")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    now = current_checksums(config, registries())
    changed = sorted(k for k, v in manifest["checksums"].items() if now.get(k) != v)
    if changed:
        raise SystemExit(f"frozen inputs changed since the freeze: {changed}. Refusing to evaluate.")
    return manifest


# ------------------------------------------------------------------ 13.2 integrity
def load_all_splits(config):
    train, val = load_development_data(config)
    test = load_split("test", allow_test=True, config=config)          # FINAL EVALUATION ONLY (after the freeze)
    cleaned = pd.read_csv(REPO_ROOT / config["dataset"]["cleaned_file"], dtype=str, keep_default_na=False)
    sha = cleaned.set_index("sample_id")["image_sha256"]
    return {s: d.assign(image_sha256=d["sample_id"].map(sha)) for s, d in (("train", train), ("validation", val), ("test", test))}


def cmd_integrity(args):
    config = load_config()
    verify_freeze(config)
    TABLES.mkdir(parents=True, exist_ok=True)
    splits = load_all_splits(config)
    a = load_split_assignments(config)                                   # manifest SHA-256 verified
    test = splits["test"]
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    check("canonical split assignment verified against its manifest (SHA-256)", True, a["split"].value_counts().to_dict())
    check("test split size = 758", len(test) == 758, len(test))
    check("test sample_ids unique", test["sample_id"].is_unique)
    for other in ("train", "validation"):
        check(f"no {other}/test sample_id overlap", not set(splits[other].sample_id) & set(test.sample_id))
        check(f"no identical image (SHA-256) shared by {other} and test",
              not set(splits[other].image_sha256) & set(test.image_sha256))
        check(f"no leakage group (Task 7 group_id) shared by {other} and test",
              not set(splits[other].group_id) & set(test.group_id))
    pairs = pd.read_csv(REPO_ROOT / config["splitting"]["near_duplicate_pairs_file"], dtype=str)
    smap = a.set_index("sample_id")["split"]
    cross = pairs[(pairs.sample_id_a.map(smap) != pairs.sample_id_b.map(smap))
                  & ((pairs.sample_id_a.map(smap) == "test") | (pairs.sample_id_b.map(smap) == "test"))]
    check("no near-duplicate image pair crosses into test (Task 5 pairs)", cross.empty, f"{len(cross)} of {len(pairs)}")
    check("test labels valid (0/1)", test["target_label"].isin(["0", "1"]).all(), test["target_label"].value_counts().to_dict())
    exists = test["image_path"].map(lambda p: (REPO_ROOT / p).is_file())
    check("all test images exist", exists.all(), f"{int(exists.sum())}/758")
    ok_sha = [hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() == s for p, s in zip(test.image_path, test.image_sha256)]
    check("test images unchanged since Task 4 (SHA-256)", all(ok_sha), f"{sum(ok_sha)}/758")
    from src.preprocessing.image import ImageLoadError, load_image
    readable = 0
    for p in test["image_path"]:
        try:
            load_image(REPO_ROOT / p)
            readable += 1
        except ImageLoadError:
            pass
    check("all test images decode with load_image()", readable == 758, f"{readable}/758")
    empty = int((test["text_raw"].str.strip() == "").sum())
    check("test caption field present for every sample (empty OCR captions counted)", test["text_raw"].notna().all(),
          f"{empty} empty captions")
    out = pd.DataFrame(checks)
    out.to_csv(TABLES / "test_integrity_checks.csv", index=False, lineterminator="\n")
    print(out.to_string(index=False))
    if not out["passed"].all():
        raise SystemExit("test integrity checks FAILED")


# ------------------------------------------------------------------ 13.3-13.5 evaluate
def cmd_evaluate(args):
    config = load_config()
    manifest = verify_freeze(config)
    regs = registries()
    names = class_names(config)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL / "metrics").mkdir(parents=True, exist_ok=True)
    test = load_all_splits(config)["test"].reset_index(drop=True)
    y = test["target_label"].astype(int).to_numpy()
    feats = Features(test, config)
    masks = subset_masks(test)
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for mid in manifest["models"]:
        pred, score, score_type = predict(mid, feats, regs)
        m = classification_metrics(y, pred, names)
        if not confusion_consistent(m, len(y)):
            raise SystemExit(f"{mid}: confusion matrix inconsistent with sample count")
        pd.DataFrame({"sample_id": test["sample_id"], "true_label": y, "predicted_label": pred,
                      "score": np.round(score, 6), "score_type": score_type}) \
            .to_csv(PRED_DIR / f"{mid}_test_predictions.csv", index=False, lineterminator="\n")
        subsets = {k: {"n": int(mask.sum()), **{kk: v for kk, v in classification_metrics(y[mask], pred[mask], names).items()
                                                 if kk in ("accuracy", "macro_f1", "weighted_f1", "precision", "recall")}}
                   for k, mask in masks.items()}
        rec = {"experiment_id": mid, "evaluation": "FINAL TEST EVALUATION", "split": "test",
               "split_assignments_sha256": manifest["checksums"]["split_assignments.csv"],
               "artifact_sha256": manifest["models"][mid]["artifact_sha256"], "role": manifest["models"][mid]["role"],
               "modality": manifest["models"][mid]["modality"], "description": manifest["models"][mid]["description"],
               "score_type": score_type, "metrics": m, "subsets": subsets,
               "validation_macro_f1_at_freeze": manifest["models"][mid]["validation_macro_f1"], "evaluated_utc": started}
        (FINAL / "metrics" / f"{mid}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  {mid}: TEST macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} (validation {manifest['models'][mid]['validation_macro_f1']:.4f})")
    # Within-model modality reliance for the final multimodal model (pre-specified diagnostic)
    e = entry_for("M04", regs)
    model, stds, _ = fusion.load_fusion(REPO_ROOT / e["artifact"])
    text = feats.finetuned("T05", config["multimodal_model"]["text_representations"]["finetuned_pooled"]["artifact"],
                           "hinglish_normalized")[0]
    image = feats.image()
    rel = {}
    for drop in ("text", "image"):
        t_in = np.repeat(stds[0].mean[None], len(y), 0) if drop == "text" else text
        i_in = np.repeat(stds[1].mean[None], len(y), 0) if drop == "image" else image
        p = fusion.predict_proba(model, stds, t_in, i_in).argmax(1)
        rel[f"{drop}_replaced_by_train_mean"] = classification_metrics(y, p, names)
    (FINAL / "metrics" / "M04_modality_reliance.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    np.savez(PRED_DIR / "test_features_for_audit.npz", sample_ids=test["sample_id"].to_numpy(str), image=image, text_t05=text)
    print("done; predictions in results/predictions/final/ (git-ignored)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--force", action="store_true", help="overwrite an existing manifest (only before any test access)")
    sub.add_parser("integrity")
    sub.add_parser("evaluate")
    args = parser.parse_args()
    {"freeze": cmd_freeze, "integrity": cmd_integrity, "evaluate": cmd_evaluate}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
