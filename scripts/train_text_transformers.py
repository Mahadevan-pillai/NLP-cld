"""Task 9: pretrained transformer text models (MuRIL), TRAIN/VALIDATION only.

Run from the repository root, step by step:
    python scripts/train_text_transformers.py tokens       # token lengths -> choose max_length
    python scripts/train_text_transformers.py frozen       # T01-T03 frozen embeddings + LR
    python scripts/train_text_transformers.py benchmark    # measure CPU fine-tuning speed
    python scripts/train_text_transformers.py finetune --exp T04   (or T05)
    python scripts/report_text_models.py                   # comparison, figures, errors, registry

TEST SET NOT EVALUATED: only load_development_data() (train + validation) is used.
Large artifacts go to $NLP_CLD_MODEL_DIR (outside the repo); small ones to models/text/.
"""

import argparse
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
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.evaluation.metrics import bootstrap_macro_f1, classification_metrics  # noqa: E402
from src.inference.text import model_dir  # noqa: E402
from src.models import transformer_text as tt  # noqa: E402
from src.models.text_baselines import preprocessing_options  # noqa: E402
from src.training.text_baselines import INPUT_COLUMN, load_development_data, subset_masks  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "text_models"
FIGURES = REPO_ROOT / "results" / "figures" / "text_models"
METRICS = REPO_ROOT / "results" / "metrics" / "text_models"
SMALL_MODELS = REPO_ROOT / "models" / "text"
MAX_LENGTH_CANDIDATES = (32, 48, 64, 96, 128)
MODE_LABEL = {"raw": "RAW", "basic": "BASIC", "hinglish_normalized": "HINGLISH_NORMALIZED"}


def setup():
    config = load_config()
    tcfg = config["text_model"]["transformer"]
    for d in (TABLES, FIGURES, METRICS, SMALL_MODELS):
        d.mkdir(parents=True, exist_ok=True)
    train, val = load_development_data(config)          # TEST is never loaded
    return config, tcfg, train, val


def revision(tcfg) -> str:
    if not tcfg.get("revision"):
        raise SystemExit("config.yaml text_model.transformer.revision is empty (pin the downloaded revision first)")
    return tcfg["revision"]


def load_tokenizer(tcfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tcfg["checkpoint"], revision=revision(tcfg))


def texts_for(df, mode, options):
    return tt.preprocess_all(df[INPUT_COLUMN].tolist(), mode, options)


# ---------------------------------------------------------------- tokens
def cmd_tokens(args):
    config, tcfg, train, val = setup()
    tok = load_tokenizer(tcfg)
    opts = preprocessing_options(config)
    rows = []
    for mode in ("raw", "basic", "hinglish_normalized"):
        for split, df in (("train", train), ("validation", val)):
            lengths = tt.token_lengths(tok, texts_for(df, mode, opts))
            row = {"variant": MODE_LABEL[mode], "split": split, "n": len(lengths),
                   **{f"p{q}": float(np.percentile(lengths, q)) for q in (50, 90, 95, 99)},
                   "max": int(lengths.max()), "mean": round(float(lengths.mean()), 2)}
            for m in MAX_LENGTH_CANDIDATES:
                row[f"truncated_pct_at_{m}"] = round(100 * tt.truncation_rate(lengths, m), 2)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "token_lengths.csv", index=False, lineterminator="\n")
    print(table.to_string(index=False))
    # Rule (decided before looking at validation): smallest candidate truncating <= 1% of TRAIN texts in every variant
    train_rows = table[table["split"] == "train"]
    choice = next(m for m in MAX_LENGTH_CANDIDATES if (train_rows[f"truncated_pct_at_{m}"] <= 1.0).all())
    print(f"\nRule: smallest max_length truncating <= 1% of TRAIN texts in all variants -> {choice}")


# ---------------------------------------------------------------- frozen
def embed_cached(tok, encoder, texts, cache: Path, max_length, batch_size):
    if cache.exists():
        return np.load(cache)
    start = time.perf_counter()
    emb = tt.mean_pooled_embeddings(texts, tok, encoder, max_length, batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    print(f"    embedded {len(texts)} texts in {time.perf_counter() - start:.0f}s -> {cache.name}")
    return emb


def cmd_frozen(args):
    from transformers import AutoModel

    config, tcfg, train, val = setup()
    names = class_names(config)
    tt.set_seed(config["random_seed"])
    tok = load_tokenizer(tcfg)
    encoder = AutoModel.from_pretrained(tcfg["checkpoint"], revision=revision(tcfg))
    opts, fz = preprocessing_options(config), tcfg["frozen"]
    cache_dir = model_dir(config).parent / "cache" / "embeddings" / f"muril_{revision(tcfg)[:8]}_len{tcfg['max_length']}_pre{opts['version']}"
    y_tr, y_va = train["target_label"].astype(int).to_numpy(), val["target_label"].astype(int).to_numpy()
    for exp, (level, mode) in tcfg["experiments"].items():
        if level != "frozen":
            continue
        start = time.perf_counter()
        x_tr = embed_cached(tok, encoder, texts_for(train, mode, opts), cache_dir / f"{mode}_train.npy",
                            tcfg["max_length"], fz["batch_size"])
        x_va = embed_cached(tok, encoder, texts_for(val, mode, opts), cache_dir / f"{mode}_validation.npy",
                            tcfg["max_length"], fz["batch_size"])
        embed_seconds = time.perf_counter() - start
        clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=config["random_seed"], **fz["logistic_regression"]))
        t0 = time.perf_counter()
        clf.fit(x_tr, y_tr)                                   # fitted on TRAIN embeddings only
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = clf.predict(x_va)
        predict_seconds = time.perf_counter() - t0
        path = SMALL_MODELS / f"{exp}_muril_frozen_logreg_{mode}_v1.joblib"
        joblib.dump(clf, path)
        reload_ok = bool(np.array_equal(joblib.load(path).predict(x_va), pred))
        write_result(exp, "frozen MuRIL + LR", mode, y_va, pred, clf.predict_proba(x_va)[:, 1], names, val, config, {
            "train_seconds": fit_seconds, "embedding_seconds_train_val": embed_seconds,
            "inference_seconds_classifier_only": predict_seconds, "embedding_dim": int(x_tr.shape[1]),
            "artifact": str(path.relative_to(REPO_ROOT).as_posix()), "artifact_bytes": path.stat().st_size,
            "reload_reproduces_predictions": reload_ok, "converged": bool(clf[-1].n_iter_[0] < clf[-1].max_iter),
            "encoder": tcfg["checkpoint"], "encoder_revision": revision(tcfg), "max_length": tcfg["max_length"],
            "classifier_settings": fz["logistic_regression"]})


def write_result(exp, model, mode, y_val, pred, scores, names, val, config, extra):
    m = classification_metrics(y_val, pred, names)
    subsets = {k: classification_metrics(y_val[mask], pred[mask], names)["macro_f1"] for k, mask in subset_masks(val).items()}
    record = {"experiment_id": exp, "model": model, "preprocessing": MODE_LABEL[mode], "preprocessing_mode": mode,
              "task": "misogyny_detection (binary)", "metrics": m, "subset_macro_f1": subsets,
              "macro_f1_ci95": bootstrap_macro_f1(y_val, pred, 1000, config["random_seed"]),
              "test_set": "NOT EVALUATED", **extra}
    (METRICS / f"{exp}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_id": val["sample_id"], "true_label": y_val, "pred": pred, "score": np.round(scores, 6)}) \
        .to_csv(METRICS / f"{exp}_validation_predictions.csv", index=False, lineterminator="\n")
    print(f"  {exp} {model:22s} {MODE_LABEL[mode]:20s} macro-F1 {m['macro_f1']:.4f} "
          f"CI {record['macro_f1_ci95']}  acc {m['accuracy']:.4f}")


# ---------------------------------------------------------------- fine-tuning
def ft_config(config, tcfg) -> tt.FineTuneConfig:
    f = tcfg["fine_tune"]
    return tt.FineTuneConfig(learning_rate=f["learning_rate"], weight_decay=f["weight_decay"],
                             batch_size=f["batch_size"], max_epochs=f["max_epochs"], warmup_ratio=f["warmup_ratio"],
                             gradient_clip_norm=f["gradient_clip_norm"],
                             early_stopping_patience=f["early_stopping_patience"],
                             freeze_word_embeddings=f["freeze_word_embeddings"], max_length=tcfg["max_length"],
                             seed=config["random_seed"])


def new_classifier(tcfg, seed):
    from transformers import AutoModelForSequenceClassification
    tt.set_seed(seed)
    return AutoModelForSequenceClassification.from_pretrained(
        tcfg["checkpoint"], revision=revision(tcfg), num_labels=2,
        id2label={0: "non_misogynistic", 1: "misogynistic"}, label2id={"non_misogynistic": 0, "misogynistic": 1})


def cmd_benchmark(args):
    config, tcfg, train, val = setup()
    tok = load_tokenizer(tcfg)
    model = new_classifier(tcfg, config["random_seed"])
    cfg = ft_config(config, tcfg)
    cfg.max_batches_per_epoch, cfg.max_epochs = 6, 1
    texts = texts_for(train, "basic", preprocessing_options(config))
    y = train["target_label"].astype(int).to_numpy()
    t0 = time.perf_counter()
    tt.fine_tune(model, tok, texts[:96], y[:96], texts[:32], y[:32], cfg, log=lambda *_: None)
    per_batch = (time.perf_counter() - t0) / 6
    t1 = time.perf_counter()
    tt.predict_logits(model, tok, texts[:256], cfg.max_length)
    infer_per_sample = (time.perf_counter() - t1) / 256
    steps = int(np.ceil(len(train) / cfg.batch_size))
    epoch_min = (per_batch * steps + infer_per_sample * len(val)) / 60
    print(f"train step (batch {cfg.batch_size}, max_len {cfg.max_length}): {per_batch:.2f}s (includes a small validation pass)")
    print(f"inference: {1000 * infer_per_sample:.1f} ms/sample")
    print(f"estimated epoch: {epoch_min:.1f} min; {tcfg['fine_tune']['max_epochs']} max epochs per run: about {epoch_min * tcfg['fine_tune']['max_epochs']:.0f} min")
    (METRICS / "finetune_benchmark.json").write_text(json.dumps({
        "seconds_per_train_step": round(per_batch, 3), "ms_per_inference_sample": round(1000 * infer_per_sample, 2),
        "estimated_minutes_per_epoch": round(epoch_min, 1), "torch_threads": torch.get_num_threads(),
        "batch_size": cfg.batch_size, "max_length": cfg.max_length}, indent=2), encoding="utf-8")


def cmd_finetune(args):
    config, tcfg, train, val = setup()
    names = class_names(config)
    level, mode = tcfg["experiments"][args.exp]
    if level != "fine_tune":
        raise SystemExit(f"{args.exp} is not a fine-tuning experiment")
    if mode == "best_non_normalized":       # decided on VALIDATION results of the frozen runs
        f1 = {m: json.loads((METRICS / f"{e}.json").read_text())["metrics"]["macro_f1"] for e, m in
              (("T01", "raw"), ("T02", "basic"))}
        mode = max(f1, key=f1.get)
        print(f"{args.exp}: best non-normalized frozen variant on validation = {mode} {f1}")
    opts = preprocessing_options(config)
    tok = load_tokenizer(tcfg)
    model = new_classifier(tcfg, config["random_seed"])
    cfg = ft_config(config, tcfg)
    tr_texts, va_texts = texts_for(train, mode, opts), texts_for(val, mode, opts)
    y_tr, y_va = train["target_label"].astype(int).to_numpy(), val["target_label"].astype(int).to_numpy()
    log_path = METRICS / f"{args.exp}_training.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(msg):
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()
        log(f"{args.exp}: fine-tuning {tcfg['checkpoint']}@{revision(tcfg)[:8]} on {MODE_LABEL[mode]}; {cfg}")
        result = tt.fine_tune(model, tok, tr_texts, y_tr, va_texts, y_va, cfg, log=log)
    pd.DataFrame([vars(h) for h in result.history]).to_csv(TABLES / f"{args.exp}_training_history.csv",
                                                          index=False, lineterminator="\n")
    t0 = time.perf_counter()
    logits = tt.predict_logits(model, tok, va_texts, cfg.max_length)
    infer = time.perf_counter() - t0
    out_dir = model_dir(config) / f"{args.exp}_muril_finetuned_{mode}_v1"
    meta = {"experiment_id": args.exp, "task": "misogyny_detection", "label_map": names,
            "base_checkpoint": tcfg["checkpoint"], "base_revision": revision(tcfg), "preprocessing_mode": mode,
            "preprocessing_version": opts["version"], "preprocessing_options": opts, "max_length": cfg.max_length,
            "fine_tune_config": vars(cfg), "best_epoch": result.best_epoch,
            "score_type": "uncalibrated softmax probability"}
    tt.save_fine_tuned(model, tok, out_dir, meta)
    model2, tok2, _ = tt.load_fine_tuned(out_dir)
    logits2 = tt.predict_logits(model2, tok2, va_texts, cfg.max_length)
    reload_ok = bool(np.allclose(logits, logits2, atol=1e-4) and np.array_equal(logits.argmax(1), logits2.argmax(1)))
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    write_result(args.exp, "fine-tuned MuRIL", mode, y_va, logits.argmax(1), tt.softmax(logits)[:, 1], names, val, config, {
        "train_seconds": result.train_seconds, "inference_seconds_validation": infer, "best_epoch": result.best_epoch,
        "epochs_run": len(result.history), "trainable_parameters": result.trainable_parameters,
        "total_parameters": result.total_parameters, "artifact": "{MODEL_DIR}/" + out_dir.name,
        "artifact_bytes": size, "reload_reproduces_predictions": reload_ok, "encoder": tcfg["checkpoint"],
        "encoder_revision": revision(tcfg), "max_length": cfg.max_length, "fine_tune_config": vars(cfg)})
    print(f"saved -> {out_dir} ({size / 1e6:.0f} MB); reload reproduces predictions: {reload_ok}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("tokens", "frozen", "benchmark"):
        sub.add_parser(name)
    ft = sub.add_parser("finetune")
    ft.add_argument("--exp", required=True, choices=["T04", "T05"])
    args = parser.parse_args()
    {"tokens": cmd_tokens, "frozen": cmd_frozen, "benchmark": cmd_benchmark, "finetune": cmd_finetune}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
