"""Create the canonical leakage-safe train/validation/test split (Task 7).

Run from the repository root:
    python scripts/create_splits.py

Inputs (read-only): Task 6 preprocessed dataset (verified against its
fingerprint), Task 4 image hashes, Task 5 near-duplicate image pairs.

Outputs:
    data/splits/split_assignments.csv   canonical sample_id -> split (tracked; IDs only)
    data/splits/split_manifest.json     reproducibility record (tracked)
    data/splits/{train,validation,test}.csv   split metadata incl. text (git-ignored)
    results/tables/splits/split_distribution.csv, leakage_audit.csv, leakage_groups.csv

The split is generated twice and compared before anything is written.
Exit code 1 on any failed blocking check.
"""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from src.data import splits as sp  # noqa: E402
from src.data.eda import class_names  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402
from src.utils.io import write_json_if_changed  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "splits"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = load_config()
    ds, cfg = config["dataset"], config["splitting"]
    seed = config["random_seed"]
    ratios = {s: cfg[s] for s in sp.SPLITS}
    key = ds["primary"]

    # ---- verify inputs ----
    input_path = REPO_ROOT / cfg["input_file"]
    fp6 = json.loads((REPO_ROOT / "results/metrics" / f"{key}_preprocessing_fingerprint.json").read_text(encoding="utf-8"))
    if not input_path.exists() or sha256(input_path) != fp6["output"]["sha256"]:
        print("Splitting aborted: Task 6 output missing or does not match its fingerprint.")
        return 1
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    cleaned = pd.read_csv(REPO_ROOT / ds["cleaned_file"], dtype=str, keep_default_na=False)
    if df["sample_id"].tolist() != cleaned["sample_id"].tolist():
        print("Splitting aborted: Task 6 and Task 4 sample IDs differ.")
        return 1
    df["image_sha256"] = cleaned["image_sha256"].values   # Task 4 hashes (re-verified in tests/pipeline)
    names = class_names(config)
    input_checks = {
        "sample_ids_unique": bool(df["sample_id"].is_unique),
        "row_count_matches_task6": len(df) == fp6["output"]["rows"],
        "labels_valid": bool(df["target_label"].astype(int).isin(names).all()),
        "text_variants_present": all(c in df for c in ("text_raw", "text_basic", "text_hinglish_normalized")),
        "images_present": bool(df["image_path"].map(lambda p: (REPO_ROOT / p).is_file()).all()),
    }
    if not all(input_checks.values()):
        print(f"Splitting aborted: input checks failed {input_checks}")
        return 1

    near = pd.read_csv(REPO_ROOT / cfg["near_duplicate_pairs_file"], dtype=str)

    # ---- groups + split (twice, for the reproducibility check) ----
    def run():
        groups, links = sp.build_groups(df, near, cfg["min_caption_words_for_grouping"])
        split = sp.assign_splits(df["target_label"].astype(int), groups, ratios, seed, cfg["n_folds"])
        return groups, links, split

    groups, links, split = run()
    groups2, _, split2 = run()
    reproducible = split.equals(split2) and groups.equals(groups2)

    # ---- audit + distribution ----
    audit = sp.leakage_audit(df, split, groups, near)
    audit = pd.concat([audit, pd.DataFrame([{"check": "reproducible_rerun_identical", "type": "blocking",
                                             "passed": reproducible, "detail": "same input/config/seed -> same split"}])],
                      ignore_index=True)
    dist = sp.distribution(df, split, names)
    gsize = groups.map(groups.value_counts())
    multi = pd.DataFrame({"group_id": groups, "sample_id": df["sample_id"], "split": split,
                          "target_label": df["target_label"]})[gsize > 1]
    group_table = multi.groupby("group_id").agg(
        size=("sample_id", "size"), split=("split", "first"), splits_spanned=("split", "nunique"),
        labels=("target_label", lambda s: ";".join(sorted(s))), members=("sample_id", lambda s: ";".join(sorted(s)))
    ).reset_index()
    reasons = links.assign(group_id=links["sample_id_a"].map(dict(zip(df["sample_id"], groups))))
    group_table["link_reasons"] = group_table["group_id"].map(
        reasons.groupby("group_id")["reason"].agg(lambda s: ";".join(sorted(set(s)))))

    TABLES.mkdir(parents=True, exist_ok=True)
    dist.to_csv(TABLES / "split_distribution.csv", index=False, lineterminator="\n")
    audit.to_csv(TABLES / "leakage_audit.csv", index=False, lineterminator="\n")
    group_table.to_csv(TABLES / "leakage_groups.csv", index=False, lineterminator="\n")

    blocking_failed = audit[(audit["type"] == "blocking") & (~audit["passed"])]
    if len(blocking_failed):
        print("Leakage audit FAILED (nothing written to data/splits):")
        print(blocking_failed.to_string(index=False))
        return 1

    # ---- canonical assignment + split files ----
    out_dir = REPO_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame({"sample_id": df["sample_id"], "split": split, "group_id": groups})
    assign_path = REPO_ROOT / cfg["assignments_file"]
    assignments.to_csv(assign_path, index=False, lineterminator="\n")
    keep = ["sample_id", "split", "group_id", "image_reference", "image_path", "original_label", "target_label",
            *ds["secondary_label_columns"], "has_text", "text_raw", "text_basic", "text_hinglish_normalized",
            "duplicate_group", "review_flags", "preprocessing_version"]
    full = df.assign(split=split, group_id=groups)[keep]
    for name in sp.SPLITS:
        full[full["split"] == name].to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")

    counts = {s: int((split == s).sum()) for s in sp.SPLITS}
    manifest = {
        "dataset": ds["candidates"][0]["name"],
        "dataset_version": ds["version"],
        "processed_dataset": {"file": cfg["input_file"], "sha256": fp6["output"]["sha256"],
                              "preprocessing_version": fp6["preprocessing_version"]},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_samples": len(df),
        "counts": counts,
        "ratios_target": ratios,
        "ratios_actual": {s: round(c / len(df), 4) for s, c in counts.items()},
        "random_seed": seed,
        "stratification": f"{cfg['stratify']} via sklearn StratifiedGroupKFold(n_splits={cfg['n_folds']}, "
                          f"shuffle=True, random_state={seed}); folds 0-2 test, 3-5 validation, 6-19 train",
        "grouping": {"method": "union-find over identical image SHA-256, Task 5 near-duplicate pairs "
                               f"(dHash <= 4), Task 4 identical-caption groups with >= "
                               f"{cfg['min_caption_words_for_grouping']} words",
                     "groups_total": int(groups.nunique()), "multi_sample_groups": int(len(group_table)),
                     "samples_in_multi_sample_groups": int(group_table["size"].sum()),
                     "largest_group": int(group_table["size"].max()) if len(group_table) else 1,
                     "mixed_label_groups": int((group_table["labels"].str.contains("0") &
                                                group_table["labels"].str.contains("1")).sum())},
        "class_distribution": dist.to_dict("records"),
        "assignments_file": cfg["assignments_file"],
        "assignments_sha256": sp.sha256_file(assign_path),
        "split_files_sha256": {s: sha256(out_dir / f"{s}.csv") for s in sp.SPLITS},
        "reproducible_rerun_identical": reproducible,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__},
        "test_set_policy": "LOCKED: final evaluation of frozen configurations only; load_split('test') "
                           "requires allow_test=True",
    }
    status = write_json_if_changed(REPO_ROOT / cfg["manifest_file"], manifest)

    print(f"Split {len(df):,} samples (seed {seed}, {cfg['n_folds']} folds):")
    for row in dist.to_dict("records"):
        print(f"  {row['split']:11s} {row['samples']:>5,} ({row['percent_of_total']:5.2f}%)  "
              f"class0 {row['percent_' + names[0]]:5.2f}%  class1 {row['percent_' + names[1]]:5.2f}%")
    print(f"  groups: {manifest['grouping']['multi_sample_groups']} multi-sample groups "
          f"({manifest['grouping']['samples_in_multi_sample_groups']} samples), largest {manifest['grouping']['largest_group']}")
    n_block = int((audit["type"] == "blocking").sum())
    print(f"  leakage audit: {n_block}/{n_block} blocking checks passed; reproducible: {reproducible}")
    print(f"  wrote {cfg['assignments_file']}, split CSVs, results/tables/splits/; manifest {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
