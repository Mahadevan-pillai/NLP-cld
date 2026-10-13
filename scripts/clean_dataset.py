"""Clean the PRIMARY dataset's metadata (raw files are only read).

Run from the repository root:
    python scripts/clean_dataset.py

Outputs:
    data/processed/<dataset>_cleaned.csv       cleaned metadata (git-ignored: contains dataset text)
    results/tables/excluded_samples.csv        every excluded sample + reason (no text)
    results/tables/duplicate_decisions.csv     every duplicate group + decision (no text)
    results/tables/cleaning_summary.csv        counts
    results/metrics/<dataset>_cleaning_fingerprint.json   reproducibility record

Deterministic: no random operations. Exit code 0 = cleaned and all
post-cleaning checks passed, 1 = failure (details printed).
"""

import csv
import hashlib
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.cleaning import CLEANING_VERSION, POLICY, clean_dataset, read_records, validate_cleaned  # noqa: E402
from src.data.datasets import DATASETS  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402
from src.utils.io import write_json_if_changed  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables"
METRICS = REPO_ROOT / "results" / "metrics"
PROCESSED = REPO_ROOT / "data" / "processed"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_sha256(path: Path) -> str:
    # Normalize line endings so the hash is the same on Windows and Linux checkouts.
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values) -> dict:
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def main() -> int:
    config = load_config()
    ds_cfg = config["dataset"]
    key = ds_cfg["primary"]
    if key not in DATASETS:
        print(f"Cleaning failed: dataset.primary={key!r} has no adapter in src/data/datasets.py")
        return 1
    spec = DATASETS[key]
    if not spec.annotation_path.exists():
        print(f"Cleaning failed: {spec.annotation_path.relative_to(REPO_ROOT)} not found "
              "(download the dataset into data/raw/ first). Nothing written.")
        return 1

    label_mapping = {str(k): int(v) for k, v in ds_cfg["label_mapping"].items()}
    raw_bytes = spec.annotation_path.read_bytes()
    raw_df, _, _ = read_records(spec.annotation_path)

    print(f"Cleaning {spec.name} ...")
    result = clean_dataset(spec, key, label_mapping, REPO_ROOT)
    cleaned = result.cleaned

    # ---- cleaned metadata (data/processed is git-ignored) ----
    out_csv = REPO_ROOT / ds_cfg["cleaned_file"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_csv, index=False, encoding="utf-8", lineterminator="\n")

    # ---- post-cleaning validation (re-read the written file) ----
    reread = pd.read_csv(out_csv, dtype=str, keep_default_na=False, encoding="utf-8")
    reread["target_label"] = reread["target_label"].astype(int)
    reread["has_text"] = reread["has_text"] == "True"
    checks = validate_cleaned(reread, label_mapping, REPO_ROOT)
    raw_text = dict(zip(raw_df["source_row"].astype(str), raw_df[spec.text_column]))
    checks["original_text_identical_to_raw"] = all(
        raw_text[r] == t for r, t in zip(reread["source_row"], reread["original_text"]))
    checks["count_reconciles"] = result.counts["retained"] + result.counts["excluded"] == result.counts["original_records"]

    # ---- tables (no dataset text in any of them) ----
    excl_fields = ["sample_id", "source_row", "image_reference", "dataset", "issue_category",
                   "action", "exclusion_reason", "duplicate_group", "kept_representative"]
    write_csv(TABLES / "excluded_samples.csv", result.excluded, excl_fields)
    dec_fields = ["group_id", "group_type", "members", "image_references", "primary_labels",
                  "decision", "rationale"]
    write_csv(TABLES / "duplicate_decisions.csv", result.duplicate_decisions, dec_fields)

    excl_by_cat = Counter(e["issue_category"] for e in result.excluded)
    flag_counts = Counter(f for flags in cleaned["review_flags"] for f in flags.split(";") if f)
    before = distribution(raw_df[spec.primary_label])
    after = distribution(cleaned["original_label"])
    summary = [("original_records", result.counts["original_records"])]
    summary += [(f"excluded_{cat}", excl_by_cat.get(cat, 0)) for cat in POLICY
                if POLICY[cat][0].startswith("REMOVE")]
    summary += [("excluded_total", result.counts["excluded"]),
                ("retained", result.counts["retained"]),
                ("retained_with_text", int(cleaned["has_text"].sum())),
                ("retained_without_text", int((~cleaned["has_text"]).sum())),
                ("retained_flagged_for_review", int((cleaned["review_flags"] != "").sum()))]
    summary += [(f"flag_{f}", n) for f, n in sorted(flag_counts.items())]
    summary += [(f"label_{spec.primary_label}={k}_before", v) for k, v in before.items()]
    summary += [(f"label_{spec.primary_label}={k}_after", v) for k, v in after.items()]
    write_csv(TABLES / "cleaning_summary.csv", [{"metric": m, "value": v} for m, v in summary],
              ["metric", "value"])

    # ---- reproducibility fingerprint ----
    image_digest = sha256_bytes("\n".join(
        f"{p}:{h}" for p, h in zip(cleaned["image_path"], cleaned["image_sha256"])).encode("utf-8"))
    fingerprint = {
        "dataset": spec.name,
        "dataset_key": key,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"citation": spec.documented.get("citation"),
                   "dataset_card": spec.documented.get("dataset_card"),
                   "license": spec.documented.get("license")},
        "input": {"annotation_file": spec.annotation_path.relative_to(REPO_ROOT).as_posix(),
                  "annotation_sha256": sha256_bytes(raw_bytes),
                  "records": result.counts["original_records"]},
        "cleaning": {"version": CLEANING_VERSION,
                     "code_sha256": code_sha256(REPO_ROOT / "src" / "data" / "cleaning.py"),
                     "label_mapping": label_mapping,
                     "random_seed": "NOT_USED (no random operations)",
                     "policy": {k: v[0] for k, v in POLICY.items()}},
        "output": {"cleaned_file": ds_cfg["cleaned_file"],
                   "cleaned_sha256": sha256_bytes(out_csv.read_bytes()),
                   "rows": len(cleaned),
                   "retained_images_digest": image_digest},
        "counts": {"excluded_by_category": dict(sorted(excl_by_cat.items())),
                   "review_flags": dict(sorted(flag_counts.items()))},
        "label_distribution": {"before": before, "after": after},
        "modality_availability": {
            "image_only_samples": len(cleaned),
            "multimodal_samples": len(cleaned),
            "text_only_samples_with_text": int(cleaned["has_text"].sum()),
            "note": "All retained samples have a readable image and a label; "
                    "text is empty for samples with has_text=False."},
        "post_cleaning_checks": checks,
    }
    status = write_json_if_changed(METRICS / f"{key}_cleaning_fingerprint.json", fingerprint)

    print(f"  records: {result.counts['original_records']}  retained: {result.counts['retained']}  "
          f"excluded: {result.counts['excluded']}  {dict(excl_by_cat)}")
    print(f"  {spec.primary_label} before: {before}  after: {after}")
    print(f"  wrote {out_csv.relative_to(REPO_ROOT).as_posix()} and results tables; fingerprint {status}")
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"Post-cleaning checks FAILED: {failed}")
        return 1
    print(f"All {len(checks)} post-cleaning checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
