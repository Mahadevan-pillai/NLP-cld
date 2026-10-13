"""Generate RAW / BASIC / HINGLISH_NORMALIZED text variants for the primary dataset.

Run from the repository root:
    python scripts/preprocess_text.py

Reads the Task 4 cleaned file (verified against its fingerprint), never
modifies it, and writes:
    data/processed/<dataset>_text_preprocessed_v<version>.csv   (git-ignored: contains text)
    results/tables/preprocessing/*.csv                          (statistics, audit; no text)
    results/tables/preprocessing/preprocessing_examples.csv      (git-ignored: contains text)
    results/metrics/<dataset>_preprocessing_fingerprint.json

Deterministic: no random operations. Exit code 1 if any validation fails.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.preprocessing import dataset as prep  # noqa: E402
from src.preprocessing import hinglish  # noqa: E402
from src.preprocessing.text import PreprocessingConfig, preprocess  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402
from src.utils.io import write_json_if_changed  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "preprocessing"


def sha256(path: Path, normalize_newlines: bool = False) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalize_newlines else data).hexdigest()


def save(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False, encoding="utf-8", lineterminator="\n")


def main() -> int:
    config = load_config()
    ds, tp = config["dataset"], config["text_preprocessing"]
    key = ds["primary"]
    cleaned_path = REPO_ROOT / ds["cleaned_file"]
    fp_path = REPO_ROOT / "results" / "metrics" / f"{key}_cleaning_fingerprint.json"
    if not cleaned_path.exists() or not fp_path.exists():
        print("Preprocessing failed: cleaned dataset or Task 4 fingerprint missing (run scripts/clean_dataset.py).")
        return 1
    cleaned_sha = sha256(cleaned_path)
    if cleaned_sha != json.loads(fp_path.read_text(encoding="utf-8"))["output"]["cleaned_sha256"]:
        print("Preprocessing aborted: cleaned dataset does not match the Task 4 fingerprint.")
        return 1

    cfg = PreprocessingConfig.from_config(config, repo_root=REPO_ROOT)
    rules = hinglish.load_rules(cfg.normalization_rules)
    df = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False, encoding="utf-8")
    print(f"Preprocessing {len(df):,} samples (preprocessing v{cfg.version}, rules v{rules.version}) ...")

    out, operations = prep.build_variants(df, cfg, ds["secondary_label_columns"])
    out_path = REPO_ROOT / tp["output_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

    # ---- validation (re-read the written file) ----
    back = pd.read_csv(out_path, dtype=str, keep_default_na=False, encoding="utf-8")
    summary = prep.summary_by_mode(out)
    checks = {
        "cleaned_input_unchanged": sha256(cleaned_path) == cleaned_sha,
        "sample_count_preserved": len(back) == len(df),
        "sample_ids_preserved": back["sample_id"].tolist() == df["sample_id"].tolist(),
        "labels_preserved": (back[["original_label", "target_label", *ds["secondary_label_columns"]]]
                             .equals(df[["original_label", "target_label", *ds["secondary_label_columns"]]])),
        "image_references_preserved": back["image_reference"].tolist() == df["image_reference"].tolist(),
        "raw_equals_original_text": back["text_raw"].tolist() == df["original_text"].tolist(),
        "no_unexpected_empty_text": int(summary["newly_empty_vs_raw"].sum()) == 0,
        "deterministic_rerun": all(
            preprocess(t, m, cfg) == v for m in ("basic", "hinglish_normalized")
            for t, v in zip(df["original_text"], out[prep.TEXT_COLUMNS[m]])),
        "single_preprocessing_version": back["preprocessing_version"].unique().tolist() == [cfg.version],
    }

    # ---- statistics, audit, examples ----
    ops_table = prep.operation_counts(operations)
    rule_table = prep.rule_frequencies(operations["hinglish_normalized"], rules)
    audit_table = prep.audit(out, operations)
    save(summary, "preprocessing_summary.csv")
    save(ops_table, "operation_counts.csv")
    save(rule_table, "hinglish_rule_frequencies.csv")
    save(audit_table, "normalization_audit.csv")
    save(prep.examples(out, operations, audit_table), "preprocessing_examples.csv")   # git-ignored

    audit_counts = {c: int(audit_table[c].sum()) for c in audit_table.columns
                    if audit_table[c].dtype == bool} if len(audit_table) else {}
    fingerprint = {
        "dataset": ds["candidates"][0]["name"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preprocessing_version": cfg.version,
        "rules_version": rules.version,
        "rules_sha256": sha256(Path(cfg.normalization_rules), normalize_newlines=True),
        "code_sha256": {p: sha256(REPO_ROOT / p, normalize_newlines=True) for p in
                        ["src/preprocessing/text.py", "src/preprocessing/hinglish.py"]},
        "config": {k: v for k, v in tp.items()},
        "input": {"file": ds["cleaned_file"], "sha256": cleaned_sha, "rows": len(df)},
        "output": {"file": tp["output_file"], "sha256": sha256(out_path), "rows": len(out),
                   "columns": list(out.columns)},
        "summary": summary.to_dict("records"),
        "rule_variants": len(rules.mapping),
        "rule_occurrences": int(rule_table["occurrences"].sum()),
        "rules_never_triggered": rule_table.loc[rule_table["occurrences"] == 0, "variant"].tolist(),
        "audit": {"flagged_samples": len(audit_table), "by_flag": audit_counts},
        "validation": checks,
        "random_seed": "NOT_USED (deterministic)",
    }
    status = write_json_if_changed(REPO_ROOT / "results" / "metrics" / f"{key}_preprocessing_fingerprint.json",
                                   fingerprint)

    for row in summary.to_dict("records"):
        print(f"  {row['mode']:20s} changed {row['changed_vs_raw']:>5,} ({row['percent_changed_vs_raw']:5.2f}%)"
              f"  vocab {row['vocabulary_eda_tokens']:,}  newly empty {row['newly_empty_vs_raw']}")
    print(f"  rule replacements: {fingerprint['rule_occurrences']:,}; audit flagged: {len(audit_table)} samples")
    print(f"  wrote {tp['output_file']} and results/tables/preprocessing/; fingerprint {status}")
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"Validation FAILED: {failed}")
        return 1
    print(f"All {len(checks)} validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
