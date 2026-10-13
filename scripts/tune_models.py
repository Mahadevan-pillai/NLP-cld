"""Task 12: compact hyperparameter tuning on VALIDATION (TEST locked).

Run from the repository root:
    python scripts/tune_models.py summary      # 12.1 pre-tuning summary of Tasks 8-11 (no re-training)
    python scripts/tune_models.py text         # A: TF-IDF + LR grid (90 trials)  -> B07
    python scripts/tune_models.py image        # C: LR / SVM C grid on cached SigLIP embeddings (16) -> I03
    python scripts/tune_models.py multimodal   # D: fusion-MLP random search (30 + baseline) + seeds -> M04
    python scripts/train_text_transformers.py finetune --exp T05 --run-id T06 --learning-rate 3e-5 --out tuning
    python scripts/train_text_transformers.py finetune --exp T05 --run-id T07 --learning-rate 5e-5 --out tuning
    python scripts/tune_models.py report       # merged trial log, best configs, before/after, figures, audit

Search spaces and rules: config.yaml `tuning`. Every search first re-runs its BASELINE configuration and
checks that it reproduces the stored Task 8/10/11 validation result exactly (same pipeline, same seed).
"""

import argparse
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
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from src.data.eda import class_names  # noqa: E402
from src.evaluation.metrics import bootstrap_macro_f1, classification_metrics  # noqa: E402
from src.models import fusion  # noqa: E402
from src.models import image_encoder as ie  # noqa: E402
from src.models.text_baselines import build_pipeline, preprocessing_options  # noqa: E402
from src.models.transformer_text import preprocess_all  # noqa: E402
from src.training import tuning  # noqa: E402
from src.training.text_baselines import load_development_data, subset_masks  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

METRICS = REPO_ROOT / "results" / "metrics" / "tuning"
TABLES = REPO_ROOT / "results" / "tables" / "tuning"
FIGURES = REPO_ROOT / "results" / "figures" / "tuning"
NAMES = None


def setup():
    global NAMES
    config = load_config()
    for d in (METRICS, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    NAMES = class_names(config)
    train, val = load_development_data(config)                  # TEST is never loaded
    return config, train, val


def scores(y_true, pred) -> dict:
    m = classification_metrics(y_true, pred, NAMES)
    return {"validation_accuracy": m["accuracy"], "validation_macro_f1": m["macro_f1"],
            "validation_weighted_f1": m["weighted_f1"]}


def write_model_result(exp: str, record: dict, val, y_va, pred, score, config) -> dict:
    """Metrics JSON + validation predictions (IDs only) for a selected tuned model."""
    m = classification_metrics(y_va, pred, NAMES)
    subsets = {k: classification_metrics(y_va[mask], pred[mask], NAMES)["macro_f1"] for k, mask in subset_masks(val).items()}
    rec = {"experiment_id": exp, "task": "misogyny_detection (binary)", **record, "metrics": m, "subset_macro_f1": subsets,
           "macro_f1_ci95": bootstrap_macro_f1(y_va, pred, 1000, config["random_seed"]), "test_set": "NOT EVALUATED"}
    (METRICS / f"{exp}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_id": val["sample_id"], "true_label": y_va, "pred": pred, "score": np.round(score, 6)}) \
        .to_csv(METRICS / f"{exp}_validation_predictions.csv", index=False, lineterminator="\n")
    print(f"  {exp}: validation macro-F1 {m['macro_f1']:.4f} CI {rec['macro_f1_ci95']}")
    return rec


# ---------------------------------------------------------------- 12.1 summary
def cmd_summary(args):
    config, _, _ = setup()
    rows = []
    b = pd.read_csv(REPO_ROOT / "results/tables/text_baselines/baseline_results.csv")
    for _, r in b.iterrows():
        rows.append({"family": "traditional text", "experiment_id": r.experiment_id, "modality": "text", "model": r.model,
                     "variant": r.preprocessing, "validation_macro_f1": r.macro_f1, "validation_accuracy": r.accuracy,
                     "macro_f1_ci95": f"{r.macro_f1_ci95_low:.4f}-{r.macro_f1_ci95_high:.4f}"})
    for family, folder, modality, ids in (("pretrained text", "text_models", "text", ["T01", "T02", "T03", "T04", "T05"]),
                                          ("image-only", "image_models", "image", ["I01", "I02"]),
                                          ("multimodal", "multimodal", "image+text", ["M01", "M02", "M03"])):
        for e in ids:
            r = json.loads((REPO_ROOT / "results/metrics" / folder / f"{e}.json").read_text(encoding="utf-8"))
            variant = r.get("preprocessing") or r.get("text_representation") or r.get("encoder", "")
            rows.append({"family": family, "experiment_id": e, "modality": modality, "model": r["model"],
                         "variant": variant, "validation_macro_f1": r["metrics"]["macro_f1"],
                         "validation_accuracy": r["metrics"]["accuracy"],
                         "macro_f1_ci95": f"{r['macro_f1_ci95'][0]:.4f}-{r['macro_f1_ci95'][1]:.4f}"})
    t = pd.DataFrame(rows)
    t["best_in_family"] = t["validation_macro_f1"] == t.groupby("family")["validation_macro_f1"].transform("max")
    t.to_csv(TABLES / "pre_tuning_summary.csv", index=False, lineterminator="\n")
    print(t[t.best_in_family].to_string(index=False))


# ---------------------------------------------------------------- A: traditional text
def tfidf_lr(params: dict, fixed: dict, token_pattern: str, seed: int):
    return make_pipeline(
        TfidfVectorizer(lowercase=True, token_pattern=token_pattern, ngram_range=tuple(params["ngram_range"]),
                        min_df=params["min_df"], max_df=fixed["max_df"], max_features=fixed["max_features"],
                        sublinear_tf=params["sublinear_tf"], strip_accents=None),
        LogisticRegression(C=params["C"], solver="lbfgs", max_iter=2000, class_weight=fixed["class_weight"],
                           random_state=seed))


def cmd_text(args):
    config, train, val = setup()
    tc, seed = config["tuning"]["traditional_text"], config["tuning"]["primary_seed"]
    base = config["text_model"]["baselines"]
    mode = "hinglish_normalized"                                   # B03's variant; preprocessing kept fixed
    opts = preprocessing_options(config)
    x_tr, x_va = preprocess_all(train.text_raw.tolist(), mode, opts), preprocess_all(val.text_raw.tolist(), mode, opts)
    y_tr, y_va = train.target_label.astype(int).to_numpy(), val.target_label.astype(int).to_numpy()
    log = tuning.TrialLog()
    baseline = {"ngram_range": list(base["tfidf"]["ngram_range"]), "min_df": base["tfidf"]["min_df"],
                "sublinear_tf": base["tfidf"]["sublinear_tf"], "C": base["logistic_regression"]["C"]}

    def trial(params):
        def fn():
            pipe = tfidf_lr(params, tc["fixed"], base["tfidf"]["token_pattern"], seed)
            t0 = time.perf_counter()
            pipe.fit(x_tr, y_tr)                                    # TF-IDF + LR fitted on TRAIN only
            secs = time.perf_counter() - t0
            return {**scores(y_va, pipe.predict(x_va)), "train_macro_f1": scores(y_tr, pipe.predict(x_tr))["validation_macro_f1"],
                    "training_time_s": round(secs, 3)}
        return fn

    b_row = log.run("TT-000", "traditional_text", "TF-IDF + LR (baseline B03)", "text", baseline, seed, trial(baseline))
    stored = pd.read_csv(REPO_ROOT / "results/tables/text_baselines/baseline_results.csv").set_index("experiment_id").loc["B03", "macro_f1"]
    if abs(b_row["validation_macro_f1"] - stored) > 1e-9:
        raise SystemExit(f"baseline check FAILED: {b_row['validation_macro_f1']} vs stored B03 {stored}")
    print(f"baseline TT-000 reproduces B03: {stored}")
    for i, params in enumerate(tuning.grid(tc["space"]), start=1):
        log.run(f"TT-{i:03d}", "traditional_text", "TF-IDF + LR", "text", params, seed, trial(params))
    log.write(TABLES / "trials_traditional_text.csv")
    best = tuning.select_best(log.rows)
    params = json.loads(best["parameters"])
    print(f"best {best['trial_id']} {params} macro-F1 {best['validation_macro_f1']}")
    # Rebuild the selected configuration as the full raw-text pipeline (TextPreprocessor inside), check, save.
    pipe = build_pipeline("logistic_regression", mode, config, seed)
    pipe.set_params(tfidf__ngram_range=tuple(params["ngram_range"]), tfidf__min_df=params["min_df"],
                    tfidf__sublinear_tf=params["sublinear_tf"], clf__C=params["C"])
    t0 = time.perf_counter()
    pipe.fit(train.text_raw.tolist(), y_tr)
    secs = time.perf_counter() - t0
    pred, score = pipe.predict(val.text_raw.tolist()), pipe.predict_proba(val.text_raw.tolist())[:, 1]
    fast = tfidf_lr(params, tc["fixed"], base["tfidf"]["token_pattern"], seed).fit(x_tr, y_tr).predict(x_va)
    path = REPO_ROOT / "models/text/B07_logistic_regression_tfidf_hinglish_normalized_tuned_v1.0.joblib"
    joblib.dump(pipe, path)
    again = joblib.load(path)
    write_model_result("B07", {
        "model": "TF-IDF + Logistic Regression (tuned)", "modality": "text", "preprocessing_mode": mode,
        "preprocessing_version": opts["version"], "base_experiment": "B03", "selected_trial": best["trial_id"],
        "parameters": params, "fixed": tc["fixed"], "random_seed": seed, "train_seconds": round(secs, 3),
        "pipeline_matches_search_predictions": bool(np.array_equal(fast, pred)),
        "reload_reproduces_predictions": bool(np.array_equal(again.predict(val.text_raw.tolist()), pred)),
        "artifact": path.relative_to(REPO_ROOT).as_posix(), "artifact_bytes": path.stat().st_size,
        "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
        "score_type": "uncalibrated logistic-regression probability of 'misogynistic'"}, val, y_va, pred, score, config)


# ---------------------------------------------------------------- C: image classifier
def cmd_image(args):
    config, train, val = setup()
    ic, seed, icfg = config["tuning"]["image"], config["tuning"]["primary_seed"], config["image_model"]
    x_tr = ie.load_cached_embeddings(config, "train", train.sample_id)      # Task 10 cache, order-checked
    x_va = ie.load_cached_embeddings(config, "validation", val.sample_id)
    y_tr, y_va = train.target_label.astype(int).to_numpy(), val.target_label.astype(int).to_numpy()
    log = tuning.TrialLog()

    def trial(kind, C):
        def fn():
            clf = ie.build_classifier(kind, {**icfg[kind], "C": C}, seed)
            t0 = time.perf_counter()
            clf.fit(x_tr, y_tr)
            secs = time.perf_counter() - t0
            return {**scores(y_va, clf.predict(x_va)), "train_macro_f1": scores(y_tr, clf.predict(x_tr))["validation_macro_f1"],
                    "training_time_s": round(secs, 3)}
        return fn

    b_row = log.run("TI-000", "image", "SigLIP + LR (baseline I01)", "image", {"classifier": "logistic_regression", "C": 1.0},
                    seed, trial("logistic_regression", 1.0))
    stored = json.loads((REPO_ROOT / "results/metrics/image_models/I01.json").read_text())["metrics"]["macro_f1"]
    if abs(b_row["validation_macro_f1"] - stored) > 1e-9:
        raise SystemExit(f"baseline check FAILED: {b_row['validation_macro_f1']} vs stored I01 {stored}")
    print(f"baseline TI-000 reproduces I01: {stored}")
    i = 0
    for kind, space in ic["space"].items():
        for C in space["C"]:
            i += 1
            log.run(f"TI-{i:03d}", "image", f"SigLIP + {kind}", "image", {"classifier": kind, "C": C}, seed, trial(kind, C))
    log.write(TABLES / "trials_image.csv")
    best = tuning.select_best(log.rows)
    params = json.loads(best["parameters"])
    print(f"best {best['trial_id']} {params} macro-F1 {best['validation_macro_f1']}")
    kind = params["classifier"]
    clf = ie.build_classifier(kind, {**icfg[kind], "C": params["C"]}, seed)
    t0 = time.perf_counter()
    clf.fit(x_tr, y_tr)
    secs = time.perf_counter() - t0
    pred, score = clf.predict(x_va), ie.classifier_scores(clf, x_va)
    path = REPO_ROOT / f"models/image/I03_siglip_{kind}_tuned_v{icfg['version']}.joblib"
    joblib.dump(clf, path)
    again = joblib.load(path)
    enc = icfg["encoder"]
    write_model_result("I03", {
        "model": f"frozen SigLIP + {kind} (tuned)", "modality": "image", "base_experiment": "I01",
        "selected_trial": best["trial_id"], "parameters": params, "classifier": kind,
        "classifier_settings": {**icfg[kind], "C": params["C"]}, "encoder": enc["checkpoint"],
        "encoder_revision": enc["revision"], "embedding": enc["embedding"], "embedding_dim": int(x_tr.shape[1]),
        "image_preprocessing": {"loader_version": icfg["preprocessing"]["loader_version"], **icfg["preprocessing"]},
        "random_seed": seed, "train_seconds": round(secs, 3), "train_macro_f1": scores(y_tr, clf.predict(x_tr))["validation_macro_f1"],
        "reload_reproduces_predictions": bool(np.array_equal(again.predict(x_va), pred) and
                                              np.allclose(ie.classifier_scores(again, x_va), score)),
        "artifact": path.relative_to(REPO_ROOT).as_posix(), "artifact_bytes": path.stat().st_size,
        "score_type": ("uncalibrated logistic-regression probability of 'misogynistic'" if kind == "logistic_regression"
                       else "decision score (signed margin, NOT a probability)")}, val, y_va, pred, score, config)


# ---------------------------------------------------------------- D: multimodal fusion MLP
def multimodal_features(config):
    """Aligned (sample_id-joined) Task 11 features via the Task 11 script's own loader."""
    spec = importlib.util.spec_from_file_location("train_multimodal", REPO_ROOT / "scripts/train_multimodal.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    _, data = tm.setup()
    feats, _ = tm.load_modalities(config, data)
    return feats


def fusion_cfg(config, params: dict, seed: int) -> fusion.FusionConfig:
    es = config["tuning"]["multimodal"]["early_stopping"]
    return fusion.FusionConfig(projection_dim=params["projection_dim"], hidden_dim=params["hidden_dim"],
                               dropout=params["dropout"], learning_rate=params["learning_rate"],
                               weight_decay=params["weight_decay"], batch_size=params["batch_size"],
                               max_epochs=es["max_epochs"], early_stopping_patience=es["patience"],
                               min_improvement=es["min_improvement"], seed=seed)


def cmd_multimodal(args):
    config, _, _ = setup()
    mc = config["tuning"]["multimodal"]
    seed0, seeds = config["tuning"]["primary_seed"], config["tuning"]["robustness_seeds"]
    feats = multimodal_features(config)
    (tr_lab, tr), (va_lab, va) = feats["train"], feats["validation"]
    y_tr, y_va = tr_lab.target_label.astype(int).to_numpy(), va_lab.target_label.astype(int).to_numpy()
    base = config["multimodal_model"]["fusion_mlp"]
    baseline = {k: base[k] for k in mc["space"]}
    log, histories, runs = tuning.TrialLog(), [], {}

    def trial(tid, params, seed, text_rep="finetuned_pooled"):
        def fn():
            cfg = fusion_cfg(config, params, seed)
            model, stds, res = fusion.train_fusion(tr[text_rep], tr["image"], y_tr, va[text_rep], va["image"], y_va, cfg,
                                                   log=lambda *_: None)
            proba = fusion.predict_proba(model, stds, va[text_rep], va["image"])
            runs[tid] = (model, stds, cfg, res, proba, text_rep)
            for h in res.history:
                histories.append({"trial_id": tid, **vars(h)})
            hist = pd.DataFrame([vars(h) for h in res.history])
            return {**scores(y_va, proba.argmax(1)),
                    "train_macro_f1": scores(y_tr, fusion.predict_proba(model, stds, tr[text_rep], tr["image"]).argmax(1))["validation_macro_f1"],
                    "epochs": len(res.history), "best_epoch": res.best_epoch, "training_time_s": round(res.train_seconds, 2),
                    "status": "EARLY_STOPPED" if len(res.history) < cfg.max_epochs else "COMPLETED",
                    "note": f"val_loss@best={hist.loc[res.best_epoch - 1, 'val_loss']:.3f}; min val_loss={hist.val_loss.min():.3f}"
                            + ("" if text_rep == "finetuned_pooled" else "; text=frozen_mean")}
        return fn

    b = log.run("TM-000", "multimodal_random_search", "fusion MLP (baseline M02)", "image+text", baseline, seed0,
                trial("TM-000", baseline, seed0))
    stored = json.loads((REPO_ROOT / "results/metrics/multimodal/M02.json").read_text())["metrics"]["macro_f1"]
    if abs(b["validation_macro_f1"] - stored) > 1e-9:
        raise SystemExit(f"baseline check FAILED: {b['validation_macro_f1']} vs stored M02 {stored}")
    print(f"baseline TM-000 reproduces M02: {stored}")
    sampled = tuning.random_configs(mc["space"], mc["n_trials"], mc["sampler_seed"], exclude=[baseline])  # baseline run above
    for i, params in enumerate(sampled, start=1):
        r = log.run(f"TM-{i:03d}", "multimodal_random_search", "fusion MLP", "image+text", params, seed0,
                    trial(f"TM-{i:03d}", params, seed0))
        print(f"  TM-{i:03d} {params} -> {r['validation_macro_f1']} ({r['status']}, best epoch {r['best_epoch']})")
    search_rows = [r for r in log.rows if r["search"] == "multimodal_random_search"]
    best = tuning.select_best(search_rows)
    best_params = json.loads(best["parameters"])
    print(f"best {best['trial_id']} {best_params} macro-F1 {best['validation_macro_f1']}")
    # Seed robustness (12.9): baseline vs best config on the same seeds; seed 42 runs are the search trials.
    seed_runs = {"baseline": {seed0: b["validation_macro_f1"]}, "tuned": {seed0: best["validation_macro_f1"]},
                 "tuned_frozen_text": {}}
    k = 0
    for label, params, rep in (("baseline", baseline, "finetuned_pooled"), ("tuned", best_params, "finetuned_pooled"),
                               ("tuned_frozen_text", best_params, "frozen_mean")):
        for s in seeds:
            if s in seed_runs[label]:
                continue
            k += 1
            r = log.run(f"TS-{k:03d}", f"seed_robustness:{label}", "fusion MLP", "image+text", params, s,
                        trial(f"TS-{k:03d}", params, s, rep))
            seed_runs[label][s] = r["validation_macro_f1"]
    summary = {lab: {**tuning.seed_summary([v[s] for s in seeds]), **{f"seed_{s}": v[s] for s in seeds}}
               for lab, v in seed_runs.items()}
    adopt = summary["tuned"]["mean"] > summary["baseline"]["mean"]
    summary["adopted"] = "tuned" if adopt else "baseline"
    summary["best_trial"], summary["best_params"] = best["trial_id"], best_params
    (METRICS / "multimodal_seed_robustness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"configuration": lab, **{k2: v for k2, v in summary[lab].items()}} for lab in seed_runs]) \
        .to_csv(TABLES / "seed_robustness.csv", index=False, lineterminator="\n")
    print(json.dumps(summary, indent=2))
    log.write(TABLES / "trials_multimodal.csv")
    pd.DataFrame(histories).to_csv(TABLES / "multimodal_trial_histories.csv", index=False, lineterminator="\n")
    # Save the adopted configuration at the PRIMARY seed (42): never the luckiest seed.
    tid = best["trial_id"] if adopt else "TM-000"
    model, stds, cfg, res, proba, rep = runs[tid]
    pred = proba.argmax(1)
    out_dir = REPO_ROOT / "models/multimodal/M04_fusion_mlp_finetuned_pooled_tuned_v1.0"
    names = NAMES
    tcfg, icfg = config["text_model"]["transformer"], config["image_model"]
    meta = {"experiment_id": "M04", "task": "misogyny_detection", "label_map": {str(k2): v for k2, v in names.items()},
            "text_representation": rep, "text_source_experiment": "T05",
            "text_encoder": f"{tcfg['checkpoint']}@{tcfg['revision']}", "text_preprocessing_mode": "hinglish_normalized",
            "text_preprocessing_version": config["text_preprocessing"]["version"],
            "image_encoder": f"{icfg['encoder']['checkpoint']}@{icfg['encoder']['revision']}",
            "image_preprocessing_version": icfg["preprocessing"]["loader_version"],
            "fusion": "projection (Linear+LayerNorm per modality) + concat + MLP", "best_epoch": res.best_epoch,
            "selected_trial": tid, "adopted": summary["adopted"], "seed": seed0,
            "score_type": "uncalibrated softmax probability of 'misogynistic'"}
    fusion.save_fusion(model, stds, cfg, out_dir, meta)
    m2, s2, _ = fusion.load_fusion(out_dir)
    proba2 = fusion.predict_proba(m2, s2, va[rep], va["image"])
    pd.DataFrame([vars(h) for h in res.history]).to_csv(TABLES / "M04_training_history.csv", index=False, lineterminator="\n")
    write_model_result("M04", {
        "model": "fusion MLP (tuned)" if adopt else "fusion MLP (baseline config kept)", "modality": "image+text",
        "base_experiment": "M02", "selected_trial": tid, "parameters": best_params if adopt else baseline,
        "fusion_config": vars(cfg), "text_representation": rep, "random_seed": seed0,
        "best_epoch": res.best_epoch, "epochs_run": len(res.history), "train_seconds": round(res.train_seconds, 2),
        "train_macro_f1": scores(y_tr, fusion.predict_proba(model, stds, tr[rep], tr["image"]).argmax(1))["validation_macro_f1"],
        "seed_robustness": summary, "reload_reproduces_predictions": bool(np.allclose(proba, proba2, atol=1e-6)),
        "artifact": out_dir.relative_to(REPO_ROOT).as_posix(),
        "artifact_bytes": sum(p.stat().st_size for p in out_dir.iterdir()),
        "text_preprocessing_mode": "hinglish_normalized", "text_preprocessing_version": config["text_preprocessing"]["version"],
        "image_preprocessing_version": icfg["preprocessing"]["loader_version"],
        "score_type": meta["score_type"]}, va_lab, y_va, pred, proba[:, 1], config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("summary", "text", "image", "multimodal", "report"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.cmd == "report":
        spec = importlib.util.spec_from_file_location("report_tuning", REPO_ROOT / "scripts/report_tuning.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main()
    {"summary": cmd_summary, "text": cmd_text, "image": cmd_image, "multimodal": cmd_multimodal}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
