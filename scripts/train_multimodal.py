"""Task 11: multimodal (image + text) fusion models, TRAIN/VALIDATION only.

Run from the repository root, step by step:
    python scripts/train_multimodal.py verify     # 11.1 inputs, artifacts, labels, usable samples
    python scripts/train_multimodal.py features   # text features (T05 pooled, frozen MuRIL) with sample_ids
    python scripts/train_multimodal.py train      # alignment audit + M01, M02, M03
    python scripts/report_multimodal.py           # ablation, comparison, disagreements, figures, registry

TEST SET NOT EVALUATED: only load_development_data() (train + validation) is used.
Modalities are joined on sample_id only (src/features/multimodal.py: align).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # models already downloaded; avoid flaky network
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.data.splits import load_split_assignments  # noqa: E402
from src.evaluation.metrics import bootstrap_macro_f1, classification_metrics  # noqa: E402
from src.features import multimodal as mm  # noqa: E402
from src.inference.text import model_dir, resolve  # noqa: E402
from src.models import fusion  # noqa: E402
from src.models import image_encoder as ie  # noqa: E402
from src.models import transformer_text as tt  # noqa: E402
from src.models.text_baselines import preprocessing_options  # noqa: E402
from src.training.text_baselines import load_development_data, subset_masks  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "multimodal"
METRICS = REPO_ROOT / "results" / "metrics" / "multimodal"
ARTIFACTS = REPO_ROOT / "models" / "multimodal"
SPLITS = ("train", "validation")


def setup():
    config = load_config()
    for d in (TABLES, METRICS, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
    train, val = load_development_data(config)          # TEST is never loaded
    cleaned = pd.read_csv(REPO_ROOT / config["dataset"]["cleaned_file"], dtype=str, keep_default_na=False)
    sha = cleaned.set_index("sample_id")["image_sha256"]
    data = {s: df.assign(image_sha256=df["sample_id"].map(sha)).reset_index(drop=True)
            for s, df in zip(SPLITS, (train, val))}
    return config, data


def text_cache_dir(config) -> Path:
    load_dotenv(REPO_ROOT / ".env", override=False)
    value = os.environ.get(config["image_model"]["cache_dir_env"], "").strip()
    rev = config["text_model"]["transformer"]["revision"][:8]
    root = Path(value).parent / "cache" / "multimodal_text" if value else REPO_ROOT / config["multimodal_model"]["cache_dir_default"]
    return root / f"muril_{rev}_pre{config['text_preprocessing']['version']}"


def text_metadata(config, rep: str, df) -> dict:
    t = config["multimodal_model"]["text_representations"][rep]
    return {"representation": rep, "source_experiment": t["source_experiment"],
            "preprocessing_mode": t["preprocessing_mode"],
            "preprocessing_version": config["text_preprocessing"]["version"],
            "encoder_revision": config["text_model"]["transformer"]["revision"],
            "max_length": config["text_model"]["transformer"]["max_length"],
            "inputs_fingerprint": mm.inputs_fingerprint(df["sample_id"], df["text_raw"])}


def image_features(config, split: str, df):
    """Task 10 cached SigLIP embeddings with their own sample_id list (verified fingerprint)."""
    cdir = ie.embedding_cache_dir(config)
    meta = json.loads((cdir / f"{split}.json").read_text(encoding="utf-8"))
    if meta["samples_fingerprint"] != ie.fingerprint(df["sample_id"], df["image_sha256"]):
        raise RuntimeError(f"image cache for {split} does not match the canonical samples / image hashes")
    return np.asarray(meta["sample_ids"]), np.load(cdir / f"{split}.npy")


# ---------------------------------------------------------------- 11.1 verify
def cmd_verify(args):
    config, data = setup()
    names = class_names(config)
    assignments = load_split_assignments(config)                    # verified against the manifest SHA-256
    checks = []

    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    counts = assignments["split"].value_counts().to_dict()
    check("canonical split unchanged (manifest SHA-256) and sizes", counts == {"train": 3532, "validation": 754, "test": 758}, counts)
    check("train and validation do not overlap", not set(data["train"].sample_id) & set(data["validation"].sample_id))
    tv = pd.concat(data.values())
    check("no identical image (SHA-256) in both train and validation",
          (tv.groupby("image_sha256")["split"].nunique() <= 1).all())
    pairs = pd.read_csv(REPO_ROOT / config["splitting"]["near_duplicate_pairs_file"], dtype=str)
    split_map = assignments.set_index("sample_id")["split"]
    check("no near-duplicate image pair crosses splits",
          (pairs["sample_id_a"].map(split_map) == pairs["sample_id_b"].map(split_map)).all())
    tcfg = config["multimodal_model"]["text_representations"]
    t05 = resolve(tcfg["finetuned_pooled"]["artifact"])
    check("Task 9 T05 checkpoint exists (weights + metadata)",
          (t05 / "model.safetensors").exists() and (t05 / "project_metadata.json").exists(), tcfg["finetuned_pooled"]["artifact"])
    check("Task 9 T03 frozen classifier exists", (REPO_ROOT / "models/text/T03_muril_frozen_logreg_hinglish_normalized_v1.joblib").exists())
    check("Task 10 I01/I02 classifiers exist", all((REPO_ROOT / f"models/image/{f}").exists() for f in
                                                   ("I01_siglip_logistic_regression_v1.0.joblib", "I02_siglip_linear_svm_v1.0.joblib")))
    for split, df in data.items():
        ids, emb = image_features(config, split, df)
        check(f"{split}: image cache covers exactly the canonical samples (ids + image hashes)",
              set(ids) == set(df.sample_id) and len(ids) == len(df), f"{len(ids)} rows, dim {emb.shape[1]}")
    val = data["validation"]
    for exp, folder in (("T05", "text_models"), ("I01", "image_models")):
        p = pd.read_csv(REPO_ROOT / "results/metrics" / folder / f"{exp}_validation_predictions.csv").set_index("sample_id")
        same = set(p.index) == set(val.sample_id) and (p.loc[val.sample_id, "true_label"].values ==
                                                         val.target_label.astype(int).values).all()
        check(f"validation labels identical to the {exp} predictions file (same ids, same labels)", same)
    usable = []
    for split, df in data.items():
        has_text = df["text_raw"].str.strip() != ""
        usable.append({"split": split, "samples": len(df), "with_image": len(df), "with_non_empty_text": int(has_text.sum()),
                       "empty_text": int((~has_text).sum()), "used_in_multimodal": len(df),
                       "misogynistic": int((df.target_label == "1").sum())})
    usable.append({"split": "test", "samples": counts["test"], "with_image": "not opened", "with_non_empty_text": "not loaded",
                   "empty_text": "not loaded", "used_in_multimodal": 0, "misogynistic": "not loaded"})
    out = pd.DataFrame(checks)
    out.to_csv(TABLES / "input_checks.csv", index=False, lineterminator="\n")
    pd.DataFrame(usable).to_csv(TABLES / "usable_samples.csv", index=False, lineterminator="\n")
    print(out.to_string(index=False))
    print(pd.DataFrame(usable).to_string(index=False))
    if not out["passed"].all():
        raise SystemExit("input checks FAILED")


# ---------------------------------------------------------------- text features
def cmd_features(args):
    from transformers import AutoModel, AutoTokenizer

    config, data = setup()
    tfc = config["text_model"]["transformer"]
    reps = config["multimodal_model"]["text_representations"]
    opts = preprocessing_options(config)
    cdir = text_cache_dir(config)
    ft_model, ft_tok, ft_meta = tt.load_fine_tuned(resolve(reps["finetuned_pooled"]["artifact"]))
    if ft_meta["preprocessing_mode"] != reps["finetuned_pooled"]["preprocessing_mode"]:
        raise SystemExit("T05 metadata preprocessing mode differs from config")
    base_tok = AutoTokenizer.from_pretrained(tfc["checkpoint"], revision=tfc["revision"])
    base = AutoModel.from_pretrained(tfc["checkpoint"], revision=tfc["revision"])
    for split, df in data.items():
        for rep in ("finetuned_pooled", "frozen_mean"):
            meta = text_metadata(config, rep, df)
            path = cdir / f"{rep}_{split}"
            if mm.load_features(path, meta) is not None:
                print(f"  {rep} {split}: cache valid")
                continue
            texts = tt.preprocess_all(df["text_raw"].tolist(), reps[rep]["preprocessing_mode"], opts)
            t0 = time.perf_counter()
            if rep == "finetuned_pooled":
                feats = mm.finetuned_pooled_features(texts, ft_model, ft_tok, tfc["max_length"])
            else:
                feats = tt.mean_pooled_embeddings(texts, base_tok, base, tfc["max_length"], tfc["frozen"]["batch_size"])
            secs = time.perf_counter() - t0
            mm.save_features(path, df["sample_id"].tolist(), feats, {**meta, "seconds": round(secs, 1)})
            print(f"  {rep} {split}: {feats.shape} in {secs:.0f}s")
    # Sanity 1: T05's own classifier head on the pooled features must reproduce the stored T05 predictions
    val = data["validation"]
    ids, feats = mm.load_features(cdir / "finetuned_pooled_validation")
    import torch
    with torch.no_grad():
        logits = ft_model.classifier(torch.from_numpy(feats)).numpy()
    stored = pd.read_csv(REPO_ROOT / "results/metrics/text_models/T05_validation_predictions.csv").set_index("sample_id").loc[ids]
    same = int((logits.argmax(1) == stored["pred"].values).sum())
    # Sanity 2: frozen features equal the Task 9 frozen cache (same order as load_development_data)
    _, frozen = mm.load_features(cdir / "frozen_mean_validation")
    old = np.load(model_dir(config).parent / "cache" / "embeddings" / f"muril_{tfc['revision'][:8]}_len{tfc['max_length']}_pre{opts['version']}" / "hinglish_normalized_validation.npy")
    diff = float(np.abs(frozen - old).max())
    sanity = {"t05_head_on_pooled_features_matches_stored_predictions": f"{same}/{len(ids)}",
              "frozen_mean_vs_task9_cache_max_abs_diff": diff}
    (METRICS / "text_feature_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    print(sanity)
    if same != len(ids) or diff > 1e-4:
        raise SystemExit("text feature sanity check FAILED")


# ---------------------------------------------------------------- training
def load_modalities(config, data):
    """Aligned features per split: {split: (labels_df, {"finetuned_pooled", "frozen_mean", "image"}, audit)}."""
    cdir, out, audit_rows = text_cache_dir(config), {}, []
    for split, df in data.items():
        mods = {}
        for rep in ("finetuned_pooled", "frozen_mean"):
            loaded = mm.load_features(cdir / f"{rep}_{split}", text_metadata(config, rep, df))
            if loaded is None:
                raise SystemExit(f"text features {rep}/{split} missing or stale: run the 'features' step")
            mods[rep] = loaded
        mods["image"] = image_features(config, split, df)
        labels, aligned, audit = mm.align(df, mods)
        for row in audit:
            audit_rows.append({"split": split, **row})
        out[split] = (labels, aligned)
    return out, pd.DataFrame(audit_rows)


def write_result(exp, record, val, y_va, pred, scores, names, config, y_tr=None, pred_tr=None):
    m = classification_metrics(y_va, pred, names)
    subsets = {k: classification_metrics(y_va[mask], pred[mask], names)["macro_f1"] for k, mask in subset_masks(val).items()}
    rec = {"experiment_id": exp, "task": "misogyny_detection (binary)", "modality": "image+text", **record,
           "metrics": m, "subset_macro_f1": subsets,
           "macro_f1_ci95": bootstrap_macro_f1(y_va, pred, 1000, config["random_seed"]),
           "train_macro_f1": classification_metrics(y_tr, pred_tr, names)["macro_f1"] if y_tr is not None else None,
           "random_seed": config["random_seed"], "test_set": "NOT EVALUATED"}
    (METRICS / f"{exp}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_id": val["sample_id"], "true_label": y_va, "pred": pred, "score": np.round(scores, 6)}) \
        .to_csv(METRICS / f"{exp}_validation_predictions.csv", index=False, lineterminator="\n")
    print(f"  {exp} {record['model']:52s} macro-F1 {m['macro_f1']:.4f} CI {rec['macro_f1_ci95']} "
          f"(train {rec['train_macro_f1']})")
    return rec


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def cmd_train(args):
    config, data = setup()
    names, seed = class_names(config), config["random_seed"]
    mcfg = config["multimodal_model"]
    feats, audit = load_modalities(config, data)
    audit.to_csv(TABLES / "alignment_audit.csv", index=False, lineterminator="\n")
    print(audit.to_string(index=False))
    (tr_lab, tr), (va_lab, va) = feats["train"], feats["validation"]
    y_tr, y_va = tr_lab["target_label"].astype(int).to_numpy(), va_lab["target_label"].astype(int).to_numpy()
    image_meta = {"image_encoder": f"{config['image_model']['encoder']['checkpoint']}@{config['image_model']['encoder']['revision']}",
                  "image_embedding": "SigLIP pooler_output (frozen, cached)", "image_dim": int(tr["image"].shape[1]),
                  "image_preprocessing_version": config["image_model"]["preprocessing"]["loader_version"]}
    histories = {}
    for exp, (kind, rep) in mcfg["experiments"].items():
        tmeta = {"text_representation": rep, "text_source_experiment": mcfg["text_representations"][rep]["source_experiment"],
                 "text_encoder": f"{config['text_model']['transformer']['checkpoint']}@{config['text_model']['transformer']['revision']}",
                 "text_preprocessing_mode": mcfg["text_representations"][rep]["preprocessing_mode"],
                 "text_preprocessing_version": config["text_preprocessing"]["version"], "text_dim": int(tr[rep].shape[1])}
        if kind == "logistic_regression":
            x_tr, x_va = np.hstack([tr[rep], tr["image"]]), np.hstack([va[rep], va["image"]])
            clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **mcfg["logistic_regression"]))
            t0 = time.perf_counter()
            clf.fit(x_tr, y_tr)                                         # TRAIN only (scaler inside)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred, scores = clf.predict(x_va), clf.predict_proba(x_va)[:, 1]
            pred_s = time.perf_counter() - t0
            path = ARTIFACTS / f"{exp}_concat_logreg_{rep}_v{mcfg['version']}.joblib"
            joblib.dump(clf, path)
            again = joblib.load(path)
            reload_ok = bool(np.array_equal(again.predict(x_va), pred) and np.allclose(again.predict_proba(x_va)[:, 1], scores))
            coef = clf[-1].coef_[0]
            write_result(exp, {"model": "early fusion: concat + StandardScaler + LR", "fusion": "concatenation (early)",
                               "classifier": "logistic_regression", "classifier_settings": mcfg["logistic_regression"],
                               **tmeta, **image_meta, "input_dim": int(x_tr.shape[1]),
                               "coef_l2_text": round(float(np.linalg.norm(coef[:tr[rep].shape[1]])), 4),
                               "coef_l2_image": round(float(np.linalg.norm(coef[tr[rep].shape[1]:])), 4),
                               "converged": bool(clf[-1].n_iter_[0] < clf[-1].max_iter),
                               "train_seconds": round(fit_s, 3), "inference_seconds_classifier_only": round(pred_s, 4),
                               "artifact": path.relative_to(REPO_ROOT).as_posix(), "artifact_bytes": path.stat().st_size,
                               "reload_reproduces_predictions": reload_ok,
                               "score_type": "uncalibrated logistic-regression probability of 'misogynistic'"},
                         va_lab, y_va, pred, scores, names, config, y_tr, clf.predict(x_tr))
        else:
            fcfg = fusion.FusionConfig(**{k: v for k, v in mcfg["fusion_mlp"].items() if k != "class_weight"}, seed=seed)
            model, stds, result = fusion.train_fusion(tr[rep], tr["image"], y_tr, va[rep], va["image"], y_va, fcfg)
            t0 = time.perf_counter()
            proba = fusion.predict_proba(model, stds, va[rep], va["image"])
            pred_s = time.perf_counter() - t0
            pred = proba.argmax(1)
            out_dir = ARTIFACTS / f"{exp}_fusion_mlp_{rep}_v{mcfg['version']}"
            meta = {"experiment_id": exp, "task": "misogyny_detection", "label_map": {str(k): v for k, v in names.items()},
                    **tmeta, **image_meta, "fusion": "projection (Linear+LayerNorm per modality) + concat + MLP",
                    "best_epoch": result.best_epoch, "score_type": "uncalibrated softmax probability of 'misogynistic'"}
            fusion.save_fusion(model, stds, fcfg, out_dir, meta)
            m2, s2, _ = fusion.load_fusion(out_dir)
            proba2 = fusion.predict_proba(m2, s2, va[rep], va["image"])
            reload_ok = bool(np.allclose(proba, proba2, atol=1e-6) and np.array_equal(proba2.argmax(1), pred))
            # Modality reliance diagnostic: replace one modality by its TRAIN mean (standardised 0) on validation
            reliance = {}
            for drop in ("text", "image"):
                t_in = np.repeat(stds[0].mean[None], len(y_va), 0) if drop == "text" else va[rep]
                i_in = np.repeat(stds[1].mean[None], len(y_va), 0) if drop == "image" else va["image"]
                reliance[f"val_macro_f1_with_{drop}_set_to_train_mean"] = classification_metrics(
                    y_va, fusion.predict_proba(model, stds, t_in, i_in).argmax(1), names)["macro_f1"]
            size = sum(p.stat().st_size for p in out_dir.iterdir())
            histories[exp] = pd.DataFrame([vars(h) for h in result.history])
            histories[exp].to_csv(TABLES / f"{exp}_training_history.csv", index=False, lineterminator="\n")
            write_result(exp, {"model": f"fusion MLP ({'fine-tuned' if rep == 'finetuned_pooled' else 'frozen'} text + image)",
                               "fusion": meta["fusion"], "classifier": "fusion_mlp", "fusion_config": vars(fcfg),
                               **tmeta, **image_meta, "parameters": result.parameters,
                               "best_epoch": result.best_epoch, "epochs_run": len(result.history),
                               "train_seconds": round(result.train_seconds, 2), "inference_seconds_classifier_only": round(pred_s, 4),
                               "artifact": out_dir.relative_to(REPO_ROOT).as_posix(), "artifact_bytes": size,
                               "weights_sha256_16": file_hash(out_dir / "fusion.safetensors"),
                               "reload_reproduces_predictions": reload_ok, **reliance,
                               "score_type": meta["score_type"]},
                         va_lab, y_va, pred, proba[:, 1], names, config, y_tr,
                         fusion.predict_proba(model, stds, tr[rep], tr["image"]).argmax(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "features", "train"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"verify": cmd_verify, "features": cmd_features, "train": cmd_train}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
