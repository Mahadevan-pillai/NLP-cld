"""Task 10: image-only models (frozen SigLIP vision tower + LR / Linear SVM), TRAIN/VALIDATION only.

Run from the repository root, step by step:
    python scripts/train_image_models.py verify   # 10.1: images exist, readable, unchanged; split checks
    python scripts/train_image_models.py embed    # frozen embeddings for train + validation (cached)
    python scripts/train_image_models.py train    # I01 (LR), I02 (Linear SVM), D01 shortcut diagnostic
    python scripts/report_image_models.py         # figures, comparison, errors, diagnostics, registry

TEST SET NOT EVALUATED: only load_development_data() (train + validation) is used,
and no test image is opened or embedded.
NO TEXT IS USED: the model input is the image only. Text columns are read only to
define the reporting subsets (with/without watermark), never as features.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # model already downloaded; avoid flaky network
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.data.splits import load_split_assignments  # noqa: E402
from src.evaluation.metrics import bootstrap_macro_f1, classification_metrics  # noqa: E402
from src.models import image_encoder as ie  # noqa: E402
from src.preprocessing.image import IMAGE_PREPROCESSING_VERSION, ImageLoadError, load_image  # noqa: E402
from src.training.text_baselines import load_development_data, subset_masks  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "image_models"
METRICS = REPO_ROOT / "results" / "metrics" / "image_models"
SMALL_MODELS = REPO_ROOT / "models" / "image"
CLASSIFIER_LABEL = {"logistic_regression": "Logistic Regression", "linear_svm": "Linear SVM"}


def setup():
    config = load_config()
    for d in (TABLES, METRICS, SMALL_MODELS):
        d.mkdir(parents=True, exist_ok=True)
    train, val = load_development_data(config)          # TEST is never loaded
    cleaned = pd.read_csv(REPO_ROOT / config["dataset"]["cleaned_file"], dtype=str, keep_default_na=False)
    sha = cleaned.set_index("sample_id")["image_sha256"]
    train, val = train.assign(image_sha256=train["sample_id"].map(sha)), val.assign(image_sha256=val["sample_id"].map(sha))
    return config, config["image_model"], train, val


def paths(df) -> list[Path]:
    # image_path only LOCATES the file; the file name (which encodes the label) is never a feature
    return [REPO_ROOT / p for p in df["image_path"]]


# ---------------------------------------------------------------- 10.1 verify
def cmd_verify(args):
    config, icfg, train, val = setup()
    assignments = load_split_assignments(config)                    # verified against the manifest SHA-256
    counts = assignments["split"].value_counts().to_dict()
    checks, rows = [], []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("canonical split counts (train/validation/test)", counts == {"train": 3532, "validation": 754, "test": 758},
          str(counts))
    check("each sample_id assigned once", assignments["sample_id"].is_unique, f"{len(assignments)} ids")
    check("train and validation do not overlap", not set(train.sample_id) & set(val.sample_id), "0 shared ids")
    check("labels valid (0/1) in train and validation",
          train.target_label.isin(["0", "1"]).all() and val.target_label.isin(["0", "1"]).all(), "")
    for split, df in (("train", train), ("validation", val)):
        for sid, p, expected_sha in zip(df["sample_id"], paths(df), df["image_sha256"]):
            row = {"sample_id": sid, "split": split, "exists": p.is_file()}
            if row["exists"]:
                raw = p.read_bytes()
                row["sha256_matches_task4"] = hashlib.sha256(raw).hexdigest() == expected_sha
                with Image.open(p) as im:
                    row["format"], row["mode"] = im.format, im.mode
                    row["width"], row["height"] = im.size
                    row["exif_orientation"] = int(im.getexif().get(274, 1))
                try:
                    load_image(p)
                    row["readable"] = True
                except ImageLoadError:
                    row["readable"] = False
            rows.append(row)
    table = pd.DataFrame(rows)
    check("all train/validation images exist", table["exists"].all(), f"{int(table['exists'].sum())}/{len(table)}")
    check("all train/validation images decode with load_image()", table["readable"].all(),
          f"{int(table['readable'].sum())}/{len(table)}")
    check("images unchanged since Task 4 (SHA-256)", table["sha256_matches_task4"].all(),
          f"{int(table['sha256_matches_task4'].sum())}/{len(table)}")
    tv = pd.concat([train, val])
    split_of_sha = tv.groupby("image_sha256")["split"].nunique()
    check("no identical image (SHA-256) in both train and validation", (split_of_sha <= 1).all(),
          f"{int((split_of_sha > 1).sum())} crossing")
    pairs = pd.read_csv(REPO_ROOT / config["splitting"]["near_duplicate_pairs_file"], dtype=str)
    split_map = assignments.set_index("sample_id")["split"]
    crossing = (pairs["sample_id_a"].map(split_map) != pairs["sample_id_b"].map(split_map)).sum()
    check("no near-duplicate image pair crosses splits (Task 5 pairs)", crossing == 0,
          f"{int(crossing)} of {len(pairs)} pairs")
    out = pd.DataFrame(checks)
    out.to_csv(TABLES / "image_data_checks.csv", index=False, lineterminator="\n")
    summary = {"train": len(train), "validation": len(val), "test_ids_counted_not_opened": counts.get("test"),
               "formats": table["format"].value_counts().to_dict(), "modes": table["mode"].value_counts().to_dict(),
               "exif_orientation_not_1": int((table["exif_orientation"] != 1).sum()),
               "min_side_px": int(table[["width", "height"]].min(axis=1).min()),
               "checks_passed": f"{int(out['passed'].sum())}/{len(out)}"}
    (METRICS / "image_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if not out["passed"].all():
        raise SystemExit("data checks FAILED")


# ---------------------------------------------------------------- embeddings
def get_embeddings(config, train, val, log=print):
    enc = config["image_model"]["encoder"]
    cdir = ie.embedding_cache_dir(config)
    result, processor, model = {}, None, None
    for split, df in (("train", train), ("validation", val)):
        meta = ie.cache_metadata(split, df["sample_id"].tolist(), df["image_sha256"].tolist(),
                                 enc["checkpoint"], enc["revision"])
        if processor is None and not ie.cache_is_valid(cdir / f"{split}.npy", cdir / f"{split}.json", meta):
            processor, model = ie.load_encoder(enc["checkpoint"], enc["revision"])
        if processor is None:
            emb, secs, cached = ie.embed_cached(paths(df), meta, cdir, None, None)
        else:
            emb, secs, cached = ie.embed_cached(paths(df), meta, cdir, processor, model, enc["batch_size"], log=log)
        if cached:
            secs = json.loads((cdir / f"{split}.json").read_text(encoding="utf-8")).get("seconds", 0.0)
        log(f"  {split}: {emb.shape} {'(cache)' if cached else ''} {secs:.0f}s")
        result[split] = (emb, secs)
    return result


def cmd_embed(args):
    config, icfg, train, val = setup()
    get_embeddings(config, train, val)
    print(f"cache: {ie.embedding_cache_dir(config)}")


def cmd_benchmark(args):
    config, icfg, train, val = setup()
    enc = icfg["encoder"]
    processor, model = ie.load_encoder(enc["checkpoint"], enc["revision"])
    sample = paths(train)[:64]
    ie.image_embeddings(sample[:8], processor, model, 8)              # warm-up
    t0 = time.perf_counter()
    ie.image_embeddings(sample, processor, model, enc["batch_size"])
    per = (time.perf_counter() - t0) / len(sample)
    print(f"{1000 * per:.0f} ms/image -> train+validation ({len(train) + len(val)}) about {per * (len(train) + len(val)) / 60:.1f} min")


# ---------------------------------------------------------------- classifiers
def write_result(exp, record, val, y_val, pred, scores, names, config):
    m = classification_metrics(y_val, pred, names)
    subsets = {k: classification_metrics(y_val[mask], pred[mask], names)["macro_f1"] for k, mask in subset_masks(val).items()}
    record = {"experiment_id": exp, "task": "misogyny_detection (binary)", "modality": "image", **record, "metrics": m,
              "subset_macro_f1": subsets, "macro_f1_ci95": bootstrap_macro_f1(y_val, pred, 1000, config["random_seed"]),
              "random_seed": config["random_seed"], "test_set": "NOT EVALUATED"}
    (METRICS / f"{exp}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_id": val["sample_id"], "true_label": y_val, "pred": pred, "score": np.round(scores, 6)}) \
        .to_csv(METRICS / f"{exp}_validation_predictions.csv", index=False, lineterminator="\n")
    print(f"  {exp} {record['model']:38s} macro-F1 {m['macro_f1']:.4f} CI {record['macro_f1_ci95']} acc {m['accuracy']:.4f}")
    return record


def resolution_features(df) -> pd.DataFrame:
    """Header-only metadata for the D01 shortcut diagnostic: exact resolution string. No pixels, no file name."""
    sizes = []
    for p in paths(df):
        with Image.open(p) as im:                                     # header only; pixels not decoded
            sizes.append(im.size)
    return pd.DataFrame({"resolution": [f"{w}x{h}" for w, h in sizes]})


def cmd_train(args):
    config, icfg, train, val = setup()
    names = class_names(config)
    enc, seed = icfg["encoder"], config["random_seed"]
    y_tr, y_va = train["target_label"].astype(int).to_numpy(), val["target_label"].astype(int).to_numpy()
    emb = get_embeddings(config, train, val)
    (x_tr, sec_tr), (x_va, sec_va) = emb["train"], emb["validation"]
    assert x_tr.shape[0] == len(train) and x_va.shape[0] == len(val)
    rows = []
    for exp, kind in icfg["experiments"].items():
        clf = ie.build_classifier(kind, icfg[kind], seed)
        t0 = time.perf_counter()
        clf.fit(x_tr, y_tr)                                          # TRAIN embeddings only
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = clf.predict(x_va)
        pred_s = time.perf_counter() - t0
        scores = ie.classifier_scores(clf, x_va)
        path = SMALL_MODELS / f"{exp}_siglip_{kind}_v{icfg['version']}.joblib"
        joblib.dump(clf, path)
        reloaded = joblib.load(path)
        reload_ok = bool(np.array_equal(reloaded.predict(x_va), pred) and np.allclose(ie.classifier_scores(reloaded, x_va), scores))
        final = clf[-1]
        converged = bool(np.max(final.n_iter_) < final.max_iter)
        rec = write_result(exp, {
            "model": f"frozen SigLIP + {CLASSIFIER_LABEL[kind]}", "encoder": enc["checkpoint"],
            "encoder_revision": enc["revision"], "classifier": kind, "classifier_settings": icfg[kind],
            "embedding": enc["embedding"], "embedding_dim": int(x_tr.shape[1]),
            "image_preprocessing": {"loader_version": IMAGE_PREPROCESSING_VERSION, **icfg["preprocessing"]},
            "score_type": ("uncalibrated logistic-regression probability of 'misogynistic'" if kind == "logistic_regression"
                           else "decision score (signed margin, NOT a probability)"),
            "feature_extraction_seconds_train_val": round(sec_tr + sec_va, 1), "train_seconds": round(fit_s, 3),
            "inference_seconds_classifier_only": round(pred_s, 4),
            "artifact": path.relative_to(REPO_ROOT).as_posix(), "artifact_bytes": path.stat().st_size,
            "reload_reproduces_predictions": reload_ok, "converged": converged, "text_features_used": False},
            val, y_va, pred, scores, names, config)
        rows.append(rec)
    # D01: shortcut diagnostic (NOT an image model): can the exact image RESOLUTION alone predict the label?
    r_tr, r_va = resolution_features(train), resolution_features(val)
    diag = make_pipeline(OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5),
                         LogisticRegression(random_state=seed, max_iter=2000))
    diag.fit(r_tr, y_tr)
    d_pred = diag.predict(r_va)
    write_result("D01", {"model": "DIAGNOSTIC: exact resolution one-hot + LR", "encoder": "none (image header only)",
                         "classifier": "logistic_regression", "embedding_dim": int(len(diag[0].get_feature_names_out())),
                         "purpose": "shortcut diagnostic for the class-correlated resolutions found in Task 5; not a model",
                         "text_features_used": False},
                 val, y_va, d_pred, diag.predict_proba(r_va)[:, 1], names, config)
    table = pd.DataFrame([{
        "experiment_id": r["experiment_id"], "encoder": f"{r['encoder']}@{r['encoder_revision'][:8]}",
        "classifier": CLASSIFIER_LABEL[r["classifier"]], "task": r["task"], "accuracy": r["metrics"]["accuracy"],
        "precision": r["metrics"]["precision"], "recall": r["metrics"]["recall"], "macro_f1": r["metrics"]["macro_f1"],
        "macro_f1_ci95_low": r["macro_f1_ci95"][0], "macro_f1_ci95_high": r["macro_f1_ci95"][1],
        "weighted_f1": r["metrics"]["weighted_f1"], "macro_f1_with_text_only": r["subset_macro_f1"]["with_text"],
        "macro_f1_without_watermark": r["subset_macro_f1"]["without_watermark"],
        "embedding_dimension": r["embedding_dim"], "feature_extraction_time_s": r["feature_extraction_seconds_train_val"],
        "training_time_s": r["train_seconds"], "inference_time_s": r["inference_seconds_classifier_only"],
        "artifact_size_bytes": r["artifact_bytes"], "random_seed": r["random_seed"]} for r in rows])
    table.to_csv(TABLES / "image_model_results.csv", index=False, lineterminator="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "embed", "train", "benchmark"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"verify": cmd_verify, "embed": cmd_embed, "train": cmd_train, "benchmark": cmd_benchmark}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
