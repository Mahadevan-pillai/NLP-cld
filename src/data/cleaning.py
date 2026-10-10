"""Deterministic, non-destructive cleaning of dataset metadata.

The raw annotation file and images are only READ. The output is a new
metadata table (one row per retained sample) plus a record of every
excluded sample and every duplicate decision.

What "cleaning" means here (Task 4): resolving genuine data-quality
problems (malformed records, missing/invalid IDs or labels, missing or
unreadable images, exact duplicate samples, conflicting labels on
identical samples). It does NOT preprocess text: `original_text` is the
raw text exactly as provided (no lowercasing, stripping, normalization).

Every decision follows POLICY below; no sample is removed silently.
"""

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.inspection import IMAGE_EXTENSIONS, check_image, dup_key

CLEANING_VERSION = "1.0"

# issue category -> (action, why). Actions: KEEP, REMOVE, REPAIR METADATA,
# FLAG FOR REVIEW (a flag is always combined with KEEP or REMOVE).
POLICY = {
    "malformed_record": ("REMOVE", "Wrong number of fields: values cannot be assigned to columns reliably."),
    "missing_id": ("REMOVE", "Without an ID the sample cannot be linked to its image or traced."),
    "duplicate_row": ("REMOVE", "Exact repeat of an earlier row: keep the first occurrence only."),
    "conflicting_duplicate_id": ("REMOVE + FLAG FOR REVIEW", "Same ID with different content: it is unknowable which row is correct."),
    "missing_label": ("REMOVE", "Supervised training/evaluation needs the primary label."),
    "invalid_label": ("REMOVE", "Primary label value outside the documented values."),
    "invalid_secondary_label": ("KEEP + FLAG FOR REVIEW", "Primary task unaffected; the sub-label value is set to empty."),
    "missing_text": ("KEEP", "A meme may contain no text; image-only and multimodal models can still use it. Marked has_text=False."),
    "image_path_repaired": ("REPAIR METADATA", "Reference differs from the file on disk only by folder/letter case; the actual file path is stored."),
    "missing_image": ("REMOVE", "The image is required for image-only and multimodal comparison on the same samples."),
    "corrupt_image": ("REMOVE", "Unreadable image cannot be used by image or multimodal models."),
    "duplicate_image_and_text": ("REMOVE", "Byte-identical image with identical text and the same primary label is the same sample: keep the first occurrence only (prevents train/test leakage)."),
    "conflicting_labels_identical_sample": ("REMOVE + FLAG FOR REVIEW", "Same image and same text annotated with different primary labels: the correct label is unknown, so all copies are excluded."),
    "same_image_different_text": ("KEEP + FLAG FOR REVIEW", "Different text makes it a different multimodal input; grouped so it stays in one split."),
    "duplicate_text_different_images": ("KEEP", "The same caption on different images can be a valid, different multimodal sample; grouped so copies stay in one split."),
    "text_label_conflict": ("KEEP + FLAG FOR REVIEW", "Same text, different images, different labels: plausible in multimodal data (image changes meaning); flagged for review."),
    "secondary_label_conflict": ("KEEP + FLAG FOR REVIEW", "Retained sample's removed duplicate had different sub-labels; exclude it from sub-label analysis."),
    "no_secondary_label": ("KEEP + FLAG FOR REVIEW", "Positive primary label but no sub-category: primary task unaffected."),
}


@dataclass
class CleaningResult:
    cleaned: pd.DataFrame
    excluded: list = field(default_factory=list)
    duplicate_decisions: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def make_sample_id(dataset_key: str, raw_id: str) -> str:
    """Stable ID from the raw ID: same input -> same ID, independent of row order.

    Hashing also hides the raw file name, which in MIMIC encodes the label.
    """
    return f"{dataset_key}_{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:12]}"


def read_records(path: Path) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Read a CSV as raw strings; separate malformed records (wrong field count)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, good, malformed = rows[0], [], []
    for number, row in enumerate(rows[1:], start=1):   # source_row = 1-based data row
        if len(row) == len(header):
            good.append([number, *row])
        else:
            malformed.append({"source_row": number, "field_count": len(row)})
    df = pd.DataFrame(good, columns=["source_row", *header])
    return df, malformed, header


def _image_index(image_root: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(image_root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}


def resolve_image(reference: str, index: dict[str, Path]) -> tuple[Path | None, bool]:
    """Return (path, repaired). Exact name match first, then basename / case-insensitive."""
    if reference in index:
        return index[reference], False
    base = reference.replace("\\", "/").rsplit("/", 1)[-1]
    if base in index:
        return index[base], True
    folded = {name.casefold(): p for name, p in index.items()}
    if base.casefold() in folded:
        return folded[base.casefold()], True
    return None, False


def clean_dataset(spec, dataset_key: str, label_mapping: dict[str, int],
                  repo_root: Path) -> CleaningResult:
    """Apply POLICY to one dataset. Deterministic: no randomness, fixed ordering."""
    df, malformed, header = read_records(spec.annotation_path)
    required = [spec.id_column, spec.primary_label, *spec.secondary_labels]
    if spec.text_column:
        required.append(spec.text_column)
    if spec.image_column:
        required.append(spec.image_column)
    missing_cols = sorted(set(required) - set(header))
    if missing_cols:
        raise ValueError(f"Annotation file is missing required columns: {missing_cols}")

    excluded, decisions = [], []
    flags = defaultdict(list)   # source_row -> review flags
    groups = {}                 # source_row -> duplicate group id
    raw_id = dict(zip(df["source_row"], df[spec.id_column]))

    def sid(row_number):
        rid = raw_id.get(row_number, "")
        return make_sample_id(dataset_key, rid) if rid.strip() else f"{dataset_key}_row{row_number}"

    def exclude(row_number, category, group="", kept=""):
        excluded.append({
            "sample_id": sid(row_number), "source_row": row_number,
            "image_reference": raw_id.get(row_number, ""), "dataset": spec.name,
            "issue_category": category, "action": POLICY[category][0],
            "exclusion_reason": POLICY[category][1], "duplicate_group": group,
            "kept_representative": kept,
        })

    for m in malformed:
        excluded.append({
            "sample_id": f"{dataset_key}_row{m['source_row']}", "source_row": m["source_row"],
            "image_reference": "", "dataset": spec.name, "issue_category": "malformed_record",
            "action": POLICY["malformed_record"][0],
            "exclusion_reason": f"{POLICY['malformed_record'][1]} ({m['field_count']} fields)",
            "duplicate_group": "", "kept_representative": "",
        })

    # 1. IDs and primary labels
    keep = []
    for _, row in df.iterrows():
        n = row["source_row"]
        if not row[spec.id_column].strip():
            exclude(n, "missing_id")
        elif not row[spec.primary_label].strip():
            exclude(n, "missing_label")
        elif row[spec.primary_label] not in label_mapping:
            exclude(n, "invalid_label")
        else:
            keep.append(n)
    df = df[df["source_row"].isin(keep)]

    # 2. Duplicate IDs: identical rows -> keep first; different content -> remove all
    content_cols = [c for c in header]
    for rid, grp in df.groupby(spec.id_column, sort=True):
        if len(grp) < 2:
            continue
        rows = grp.sort_values("source_row")
        if (rows[content_cols].nunique() <= 1).all():
            for n in rows["source_row"].iloc[1:]:
                exclude(n, "duplicate_row", kept=sid(rows["source_row"].iloc[0]))
        else:
            for n in rows["source_row"]:
                exclude(n, "conflicting_duplicate_id")
    removed = {e["source_row"] for e in excluded}
    df = df[~df["source_row"].isin(removed)]

    # 3. Secondary labels: invalid values are flagged, primary task unaffected
    secondary = {}
    for _, row in df.iterrows():
        values = {}
        for col in spec.secondary_labels:
            if row[col] in label_mapping:
                values[col] = label_mapping[row[col]]
            else:
                values[col] = pd.NA
                flags[row["source_row"]].append("invalid_secondary_label")
        secondary[row["source_row"]] = values

    # 4. Images: resolve path, check readability, hash
    image_info = {}
    if spec.image_column:
        index = _image_index(spec.image_root)
        for _, row in df.iterrows():
            n = row["source_row"]
            path, repaired = resolve_image(row[spec.image_column], index)
            if path is None:
                exclude(n, "missing_image")
                continue
            info = check_image(path)
            if not info["readable"]:
                exclude(n, "corrupt_image")
                continue
            if repaired:
                flags[n].append("image_path_repaired")
            image_info[n] = {"path": path.relative_to(repo_root).as_posix(), "sha256": info["sha256"]}
        df = df[df["source_row"].isin(image_info)]

    text_key = (lambda r: dup_key(r[spec.text_column])) if spec.text_column else (lambda r: "")

    # 5. Byte-identical images
    by_hash = defaultdict(list)
    for _, row in df.iterrows():
        by_hash[image_info[row["source_row"]]["sha256"]].append(row)
    for digest, rows in sorted(by_hash.items()):
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r["source_row"])
        group = f"img_{digest[:10]}"
        members = [sid(r["source_row"]) for r in rows]
        labels = [r[spec.primary_label] for r in rows]
        base = {"group_id": group, "members": ";".join(members),
                "image_references": ";".join(r[spec.image_column] for r in rows),
                "primary_labels": ";".join(labels)}
        if len({text_key(r) for r in rows}) > 1:
            for r in rows:
                flags[r["source_row"]].append("same_image_different_text")
                groups[r["source_row"]] = group
            decisions.append({**base, "group_type": "identical_image_different_text",
                              "decision": "KEEP ALL (grouped, flagged)",
                              "rationale": POLICY["same_image_different_text"][1]})
        elif len(set(labels)) > 1:
            for r in rows:
                exclude(r["source_row"], "conflicting_labels_identical_sample", group=group)
            decisions.append({**base, "group_type": "identical_image_and_text",
                              "decision": "REMOVE ALL (flag for review)",
                              "rationale": POLICY["conflicting_labels_identical_sample"][1]})
        else:
            first = rows[0]["source_row"]
            for r in rows[1:]:
                exclude(r["source_row"], "duplicate_image_and_text", group=group, kept=sid(first))
            sub_differs = any(secondary[r["source_row"]] != secondary[first] for r in rows[1:])
            if sub_differs:
                flags[first].append("secondary_label_conflict")
            decisions.append({**base, "group_type": "identical_image_and_text",
                              "decision": f"KEEP FIRST ({sid(first)}), REMOVE {len(rows) - 1}"
                                          + (" (sub-labels differ: flagged)" if sub_differs else ""),
                              "rationale": POLICY["duplicate_image_and_text"][1]})
    removed = {e["source_row"] for e in excluded}
    df = df[~df["source_row"].isin(removed)]

    # 6. Duplicate text on different images (kept, grouped for split-time leakage control)
    if spec.text_column:
        has_text = df[spec.text_column].str.strip() != ""
        keyed = df[has_text].assign(_key=df.loc[has_text, spec.text_column].map(dup_key))
        for key, grp in sorted(keyed.groupby("_key"), key=lambda kv: kv[1]["source_row"].min()):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("source_row")
            group = f"txt_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:10]}"
            conflict = grp[spec.primary_label].nunique() > 1
            for n in grp["source_row"]:
                groups.setdefault(n, group)
                flags[n].append("duplicate_text")
                if conflict:
                    flags[n].append("text_label_conflict")
            decisions.append({
                "group_id": group, "group_type": "duplicate_text_different_images",
                "members": ";".join(sid(n) for n in grp["source_row"]),
                "image_references": ";".join(grp[spec.image_column]) if spec.image_column else "",
                "primary_labels": ";".join(grp[spec.primary_label]),
                "decision": "KEEP ALL (grouped" + (", flagged: labels differ)" if conflict else ")"),
                "rationale": POLICY["text_label_conflict" if conflict else "duplicate_text_different_images"][1],
            })

    # 7. Positive primary label without any sub-category
    if spec.secondary_labels and spec.positive_value is not None:
        positive = label_mapping[spec.positive_value]
        for _, row in df.iterrows():
            n = row["source_row"]
            if label_mapping[row[spec.primary_label]] == positive and \
                    not any(v == positive for v in secondary[n].values() if v is not pd.NA):
                flags[n].append("no_secondary_label")

    # 8. Output table (raw values preserved next to internal ones)
    out = pd.DataFrame({
        "sample_id": [sid(n) for n in df["source_row"]],
        "source_row": df["source_row"].astype(int).values,
        "image_reference": df[spec.image_column].values if spec.image_column else "",
        "image_path": [image_info[n]["path"] for n in df["source_row"]] if spec.image_column else "",
        "image_sha256": [image_info[n]["sha256"] for n in df["source_row"]] if spec.image_column else "",
        "original_text": df[spec.text_column].values if spec.text_column else "",
        "has_text": (df[spec.text_column].str.strip() != "").values if spec.text_column else False,
        "original_label": df[spec.primary_label].values,
        "target_label": df[spec.primary_label].map(label_mapping).astype(int).values,
    })
    for col in spec.secondary_labels:
        out[col] = pd.array([secondary[n][col] for n in df["source_row"]], dtype="Int64")
    out["duplicate_group"] = [groups.get(n, "") for n in df["source_row"]]
    out["review_flags"] = [";".join(sorted(set(flags[n]))) for n in df["source_row"]]
    out = out.sort_values("source_row").reset_index(drop=True)

    if out["sample_id"].duplicated().any():
        raise RuntimeError("sample_id collision: cannot produce unique stable IDs")

    excluded.sort(key=lambda e: (e["source_row"], e["issue_category"]))
    decisions.sort(key=lambda d: (d["group_type"], d["group_id"]))
    counts = {"original_records": len(malformed) + len(raw_id), "retained": len(out),
              "excluded": len(excluded)}
    return CleaningResult(out, excluded, decisions, counts)


def validate_cleaned(cleaned: pd.DataFrame, label_mapping: dict[str, int], repo_root: Path,
                     check_images: bool = True) -> dict:
    """Post-cleaning integrity checks. Returns {check: bool} plus details."""
    valid_targets = set(label_mapping.values())
    checks = {
        "sample_ids_unique": not cleaned["sample_id"].duplicated().any(),
        "sample_ids_present": bool(cleaned["sample_id"].astype(str).str.strip().ne("").all()),
        "target_labels_valid": bool(cleaned["target_label"].isin(valid_targets).all()),
        "original_label_matches_target": bool(
            (cleaned["original_label"].map(label_mapping) == cleaned["target_label"]).all()),
        "image_paths_unique": not cleaned["image_path"].duplicated().any(),
        "no_unflagged_identical_images": not cleaned.loc[
            ~cleaned["review_flags"].str.contains("same_image_different_text"), "image_sha256"
        ].duplicated().any(),
        "has_text_consistent": bool(
            (cleaned["has_text"] == (cleaned["original_text"].str.strip() != "")).all()),
    }
    if check_images:
        unreadable = [p for p in cleaned["image_path"]
                      if not check_image(repo_root / p)["readable"]]
        checks["all_images_readable"] = not unreadable
    return checks
