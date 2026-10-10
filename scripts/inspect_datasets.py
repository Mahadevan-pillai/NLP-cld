"""Inspect every dataset under data/raw/ (read-only) and write summaries.

Run from the repository root:
    python scripts/inspect_datasets.py

Outputs (small, contain no dataset text, safe to version):
    results/metrics/dataset_inspection_<name>.json   full measurements
    results/tables/dataset_inventory.csv              one row per dataset
    results/tables/dataset_compatibility.csv          role / suitability

Re-running with the same raw files produces the same measurements
(only "generated_at" changes). Nothing under data/raw is modified.
"""

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import inspection as ins  # noqa: E402
from src.data.datasets import DATASETS, RAW_DIR  # noqa: E402
from src.utils.config import REPO_ROOT  # noqa: E402
from src.utils.io import write_json_if_changed  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables"
METRICS = REPO_ROOT / "results" / "metrics"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(spec) -> dict:
    if not spec.annotation_path.exists():
        return {"dataset": spec.name, "status": "ANNOTATION_FILE_MISSING"}

    df = ins.read_table(spec.annotation_path)
    report = {
        "dataset": spec.name,
        "status": "INSPECTED",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotation_file": f"data/raw/{spec.directory}/{spec.annotation_file}",
        "annotation_sha256": sha256_file(spec.annotation_path),
        "documented": spec.documented,
        "recommended_role": spec.recommended_role,
        "files": ins.file_inventory(spec.root),
        "schema": ins.schema_summary(df),
        "column_roles": {
            "id": spec.id_column, "text": spec.text_column, "image_reference": spec.image_column,
            "primary_label": spec.primary_label, "secondary_labels": spec.secondary_labels,
        },
        "labels": {c: ins.label_summary(df[c]) for c in [spec.primary_label, *spec.secondary_labels]},
    }
    if spec.secondary_labels and spec.positive_value is not None:
        report["sublabel_consistency"] = ins.sublabel_consistency(
            df, spec.primary_label, spec.secondary_labels, spec.positive_value, spec.id_column)
    if spec.text_column:
        report["text"] = ins.text_summary(df[spec.text_column])
        empty = df[spec.text_column].str.strip() == ""
        report["text"]["missing_or_empty_by_primary_label"] = {
            str(k): int(v) for k, v in df.loc[empty, spec.primary_label].value_counts().sort_index().items()}
        report["text"]["ids_missing_or_empty"] = sorted(df.loc[empty, spec.id_column])
    report["duplicates"] = ins.duplicate_summary(df, spec.id_column, spec.text_column, spec.primary_label)
    if spec.image_column:
        report["filename_label_association"] = ins.filename_label_association(
            df[spec.image_column], df[spec.primary_label])
        images = ins.image_summary(spec.root, df[spec.image_column])
        # Exact-duplicate images whose rows carry different primary labels.
        label_of = dict(zip(df[spec.image_column], df[spec.primary_label]))
        groups = images.get("exact_duplicate_image_names", [])
        images["duplicate_image_groups_with_conflicting_labels"] = sum(
            len({label_of.get(n) for n in g}) > 1 for g in groups)
        report["images"] = images
    return report


def unregistered_items() -> list[str]:
    """Anything in data/raw that no DatasetSpec covers (reported, not skipped)."""
    known = {spec.directory for spec in DATASETS.values()}
    return sorted(p.name for p in RAW_DIR.iterdir() if p.name != ".gitkeep" and p.name not in known)


def measured_issues(r: dict) -> list[str]:
    """Problems derived from the measurements (nothing hard-coded per dataset)."""
    d, dup, img = r["documented"], r["duplicates"], r.get("images", {})
    issues = []
    if "TO_BE_VERIFIED" in d.get("label_value_meaning", "TO_BE_VERIFIED"):
        issues.append("label value meaning TO_BE_VERIFIED")
    if r.get("filename_label_association", {}).get("pattern_perfectly_predicts_label"):
        issues.append("file names encode the label: never use file names as features")
    if dup.get("duplicate_text_groups_with_conflicting_labels"):
        issues.append(f"{dup['duplicate_text_groups_with_conflicting_labels']} duplicate-text groups have conflicting labels")
    if img.get("duplicate_image_groups_with_conflicting_labels"):
        issues.append(f"{img['duplicate_image_groups_with_conflicting_labels']} identical-image groups have conflicting labels")
    if dup.get("rows_in_case_whitespace_duplicate_groups") or img.get("exact_duplicate_image_groups"):
        issues.append("duplicate text/images must be kept within one split (leakage)")
    if "not present in the local file" in d.get("reported_split", ""):
        issues.append("official split not in local file")
    return issues


def inventory_row(r: dict) -> dict:
    d, labels, dup = r["documented"], r["labels"], r["duplicates"]
    primary = r["column_roles"]["primary_label"]
    img, txt = r.get("images", {}), r.get("text", {})
    return {
        "dataset_name": r["dataset"],
        "version": f"{Path(r['annotation_file']).name} (sha256 {r['annotation_sha256'][:12]})",
        "primary_task": d["tasks"],
        "languages": f"{d['language']} (documented; heuristic check in report)",
        "modalities": d["modalities"],
        "sample_count": r["schema"]["rows"],
        "image_count": img.get("found", 0),
        "text_count": txt.get("non_empty", 0),
        "primary_label": primary,
        "number_of_classes": len(labels[primary]["unique_values"]),
        "missing_text": txt.get("missing_or_empty", "NOT_APPLICABLE"),
        "missing_images": img.get("missing", "NOT_APPLICABLE"),
        "corrupt_images": img.get("corrupt_or_unreadable", "UNKNOWN"),
        "duplicate_ids": dup["duplicate_ids"],
        "duplicate_text_rows": dup.get("rows_in_case_whitespace_duplicate_groups", "NOT_APPLICABLE"),
        "duplicate_text_conflicting_label_groups": dup.get("duplicate_text_groups_with_conflicting_labels", "NOT_APPLICABLE"),
        "exact_duplicate_image_groups": img.get("exact_duplicate_image_groups", "UNKNOWN"),
        "license_status": d["license"],
        "source_status": ("Paper DOI + dataset card found; confirm your download source matches"
                          if d.get("dataset_card") else "Paper DOI verified; download source TO_BE_VERIFIED"),
        "notes": "; ".join(measured_issues(r)),
    }


def compatibility_row(r: dict) -> dict:
    labels, img, txt = r["labels"], r.get("images", {}), r.get("text", {})
    images_ok = img.get("status") == "INSPECTED" and img.get("missing", 1) == 0
    text_ok = txt.get("non_empty", 0) > 0
    primary = r["column_roles"]["primary_label"]
    return {
        "dataset": r["dataset"],
        "primary_task": r["documented"]["tasks"],
        "task_scope": r["documented"].get("task_scope", "TO_BE_VERIFIED"),
        "languages": r["documented"]["language"],
        "text_availability": f"{txt.get('non_empty', 0)}/{r['schema']['rows']} non-empty",
        "image_availability": f"{img.get('found', 0)}/{img.get('unique_referenced_names', 0)} found, "
                              f"{img.get('corrupt_or_unreadable', 'n/a')} unreadable",
        "labels": f"{primary} " + "/".join(labels[primary]["unique_values"])
                  + "; sub-labels: " + ", ".join(r["column_roles"]["secondary_labels"]),
        "classification_type": "binary primary label + multi-label sub-categories",
        "sample_count": r["schema"]["rows"],
        "text_model_suitability": "SUITABLE" if text_ok else "NOT_SUITABLE",
        "image_model_suitability": "SUITABLE" if images_ok else "BLOCKED (images incomplete)",
        "multimodal_model_suitability": "SUITABLE (same samples and labels for all three models)"
                                        if images_ok and text_ok else "BLOCKED",
        "potential_project_role": r["recommended_role"],
        "compatibility_issues": "; ".join(
            ([] if len(DATASETS) > 1 else ["only dataset available: no merge question arises"])
            + measured_issues(r)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(out: Path, report: dict) -> str:
    return write_json_if_changed(out, report)


def main() -> int:
    reports = []
    for key, spec in DATASETS.items():
        print(f"Inspecting {spec.name} ...")
        report = inspect(spec)
        if report["status"] != "INSPECTED":
            # Never overwrite an existing report with a "missing data" stub
            # (e.g. on a machine where the dataset is not downloaded).
            print(f"  status: {report['status']} (no report written)")
            continue
        out = METRICS / f"dataset_inspection_{key}.json"
        print(f"  status: INSPECTED -> {out.relative_to(REPO_ROOT)} ({write_report(out, report)})")
        reports.append(report)

    extra = unregistered_items()
    if extra:
        print(f"WARNING: items in data/raw without an adapter (not inspected): {extra}")

    if reports:
        write_csv(TABLES / "dataset_inventory.csv", [inventory_row(r) for r in reports])
        write_csv(TABLES / "dataset_compatibility.csv", [compatibility_row(r) for r in reports])
        print("Wrote results/tables/dataset_inventory.csv and dataset_compatibility.csv")
    return 0 if reports else 1


if __name__ == "__main__":
    sys.exit(main())
