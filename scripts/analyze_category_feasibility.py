"""Multi-category / hierarchical label feasibility analysis (Task 8.5).

Run from the repository root:
    python scripts/analyze_category_feasibility.py

Reads only verified annotations (MIMIC cleaned file + canonical split
assignment). Trains nothing, changes no labels, touches no Task 8 output.
Label counts per split are descriptive statistics only; no model is
evaluated on any split here.

Writes results/tables/category_feasibility/:
    all_dataset_labels.csv, label_mapping_analysis.csv,
    category_sufficiency.csv, subcategory_structure.csv, modality_support.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.categories import feasibility_tier, load_categories  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

OUT = REPO_ROOT / "results" / "tables" / "category_feasibility"
CONCEPTS = ["SAFE", "MISOGYNY", "HATE", "ABUSE", "THREAT", "OBSCENE", "SARCASM"]

# Other files on the developer machine, inspected read-only on 2026-10-12
# (Downloads folder, not part of the project). Recorded here because the
# files live outside the repository.
EXTERNAL_FILES = [
    {"dataset": "DATASET 00.csv (sha256 dc0beb6901e7)", "domain": "agriculture (crop, soil, moisture, temperature)",
     "label_column": "result", "unique_labels": "0;1;2", "counts": "0:9062; 1:6227; 2:1122", "rows": 16411},
    {"dataset": "DATASET 01.csv (sha256 59e8a76ac5ce)", "domain": "mortality statistics by country/age/sex",
     "label_column": "(none: numeric statistics; 'Sex' is a demographic field)", "unique_labels": "Male;Female;Both",
     "counts": "19635 each", "rows": 58905},
    {"dataset": "DATASET 02.xlsx (sha256 dbad641c16c9)", "domain": "FIFA World Cup match results (3 sheets)",
     "label_column": "(none)", "unique_labels": "", "counts": "", "rows": "853 / 1705 / 21"},
]

# Mapping judgements: (label, concept) -> (status, confidence, evidence, semantic difference)
MAPPING = {
    ("Misogyny=1", "MISOGYNY"): ("SUPPORTED", "high", "Dataset card: 1 = Misogynistic; binary task of MIMIC.",
                                 "Annotated as misogyny in memes (image + text), Hindi-English context."),
    ("Misogyny=0", "SAFE"): ("NOT_SUPPORTED", "high", "0 = 'Non Misogynistic' (dataset card). Other harms were never annotated.",
                             "Non-misogynistic memes may still contain hate, insults, threats or obscenity."),
    ("Objectification", "OBSCENE"): ("AMBIGUOUS", "low", "Sub-label of misogyny; no definition of obscenity/sexual explicitness in the release.",
                                     "Objectification (treating women as objects) is not the same as obscene/sexual content; it is also only annotated within misogynistic memes."),
    ("Objectification", "MISOGYNY"): ("SUPPORTED", "high", "Sub-category of misogyny (dataset card, sub-task 2).",
                                      "A misogyny subcategory, not a separate harm category."),
    ("Prejudice", "HATE"): ("PARTIALLY_SUPPORTED", "low", "Gender prejudice is one form of identity-based hostility.",
                            "Covers only prejudice against women, only within misogynistic memes; cannot represent hate against other groups (religion, caste, ethnicity...)."),
    ("Prejudice", "MISOGYNY"): ("SUPPORTED", "high", "Sub-category of misogyny (dataset card, sub-task 2).",
                                "A misogyny subcategory."),
    ("Humiliation", "ABUSE"): ("PARTIALLY_SUPPORTED", "low", "Humiliation overlaps with insulting/degrading content.",
                               "Covers humiliation of women only, within misogynistic memes; not general personal insult/abuse."),
    ("Humiliation", "MISOGYNY"): ("SUPPORTED", "high", "Sub-category of misogyny (dataset card, sub-task 2).",
                                  "A misogyny subcategory."),
}


def label_inventory(df: pd.DataFrame, raw: pd.DataFrame, subs: list[str]) -> pd.DataFrame:
    rows = [{
        "dataset": "MIMIC", "label_column": "Misogyny", "role": "primary", "task_type": "binary",
        "unique_labels": "0;1", "documented_meaning": "1 = Misogynistic, 0 = Non Misogynistic (dataset card)",
        "counts_cleaned": "; ".join(f"{k}:{v}" for k, v in df["original_label"].value_counts().sort_index().items()),
        "counts_raw": "; ".join(f"{k}:{v}" for k, v in raw["Misogyny"].value_counts().sort_index().items()),
        "labels_can_coexist": "no (single binary label)", "missing_labels": int((df["original_label"] == "").sum()),
        "content_moderation_label": True}]
    for c in subs:
        rows.append({
            "dataset": "MIMIC", "label_column": c, "role": "secondary (sub-category of misogyny)", "task_type": "binary per label (multi-label set)",
            "unique_labels": "0;1", "documented_meaning": f"{c} category of a misogynistic meme (dataset card, sub-task 2); value meaning 1 = present (structurally consistent: 0 positives among non-misogynistic)",
            "counts_cleaned": "; ".join(f"{k}:{v}" for k, v in df[c].value_counts().sort_index().items()),
            "counts_raw": "; ".join(f"{k}:{v}" for k, v in raw[c].value_counts().sort_index().items()),
            "labels_can_coexist": "yes (with other sub-labels)", "missing_labels": int((df[c] == "").sum()),
            "content_moderation_label": True})
    for e in EXTERNAL_FILES:
        rows.append({"dataset": e["dataset"], "label_column": e["label_column"], "role": "not a content dataset",
                     "task_type": "n/a", "unique_labels": e["unique_labels"], "documented_meaning": e["domain"],
                     "counts_cleaned": "", "counts_raw": e["counts"], "labels_can_coexist": "n/a",
                     "missing_labels": "", "content_moderation_label": False})
    return pd.DataFrame(rows)


def mapping_table(subs: list[str]) -> pd.DataFrame:
    rows = []
    for label in ["Misogyny=1", "Misogyny=0", *subs]:
        for concept in CONCEPTS:
            status, conf, evidence, diff = MAPPING.get((label, concept), (
                "NOT_SUPPORTED", "high", "No annotation in MIMIC corresponds to this concept.", "-"))
            rows.append({"source_dataset": "MIMIC", "original_label": label, "proposed_concept": concept,
                         "status": status, "confidence": conf, "evidence": evidence, "semantic_difference": diff})
    for e in EXTERNAL_FILES:
        rows.append({"source_dataset": e["dataset"], "original_label": e["label_column"], "proposed_concept": "ALL",
                     "status": "NOT_SUPPORTED", "confidence": "high",
                     "evidence": f"Unrelated domain: {e['domain']}.", "semantic_difference": "-"})
    return pd.DataFrame(rows)


def sufficiency(df: pd.DataFrame, subs: list[str]) -> pd.DataFrame:
    rows = []
    mis = df["original_label"] == "1"
    specs = [("MISOGYNY", "binary (all samples)", df["original_label"] == "1", pd.Series(True, index=df.index))]
    specs += [(c.upper(), "multilabel, conditional on misogyny", df[c] == "1", mis) for c in subs]
    for name, task, positive, population in specs:
        pos = positive & population
        row = {"category": name, "task": task, "population": int(population.sum()),
               "positives": int(pos.sum()), "negatives": int((population & ~positive).sum()),
               "percent_positive_in_population": round(100 * pos.sum() / population.sum(), 2),
               "percent_positive_all_samples": round(100 * pos.sum() / len(df), 2),
               "imbalance_ratio_neg_to_pos": round((population & ~positive).sum() / pos.sum(), 2)}
        for split in ("train", "validation", "test"):
            row[f"positives_{split}"] = int((pos & (df["split"] == split)).sum())
        row["feasibility"] = feasibility_tier(row["positives_train"], row["positives_validation"])
        rows.append(row)
    for concept in ("SAFE", "HATE", "ABUSE", "THREAT", "OBSCENE", "SARCASM"):
        rows.append({"category": concept, "task": "not annotated", "population": 0, "positives": 0,
                     "feasibility": "INSUFFICIENT (no annotation)"})
    table = pd.DataFrame(rows)
    for col in ("negatives", "positives_train", "positives_validation", "positives_test"):
        table[col] = table[col].astype("Int64")   # keep integers despite empty unsupported rows
    return table


def subcategory_structure(df: pd.DataFrame, subs: list[str]) -> pd.DataFrame:
    pos = df[df["original_label"] == "1"]
    combo = pos[subs].apply(lambda r: "+".join(c for c in subs if r[c] == "1") or "(none)", axis=1)
    table = combo.value_counts().rename_axis("combination").reset_index(name="misogynistic_samples")
    table["percent"] = (100 * table["misogynistic_samples"] / len(pos)).round(2)
    table["n_labels"] = table["combination"].map(lambda c: 0 if c == "(none)" else c.count("+") + 1)
    return table


def modality_support(df: pd.DataFrame, suff: pd.DataFrame) -> pd.DataFrame:
    has_text = int((df["has_text"] == "True").sum())
    rows = []
    for _, r in suff[suff["population"] > 0].iterrows():
        rows.append({"category": r["category"], "dataset": "MIMIC", "text_supported": True, "image_supported": True,
                     "multimodal_supported": True, "sample_count": int(r["population"]),
                     "positives": int(r["positives"]), "samples_with_text_in_population": int(
                         ((df["has_text"] == "True") & (df["original_label"] == "1" if r["category"] != "MISOGYNY"
                                                         else True)).sum()),
                     "recommended_model_task": "stage 1 binary" if r["category"] == "MISOGYNY"
                     else "stage 2 multilabel head (conditional on misogyny)",
                     "note": f"every sample has image + OCR text field; {len(df) - has_text} samples have empty text"})
    for concept in ("SAFE", "HATE", "ABUSE", "THREAT", "OBSCENE", "SARCASM"):
        rows.append({"category": concept, "dataset": "none", "text_supported": False, "image_supported": False,
                     "multimodal_supported": False, "sample_count": 0, "positives": 0,
                     "samples_with_text_in_population": 0, "recommended_model_task": "NOT TRAINABLE (no annotation)",
                     "note": ""})
    return pd.DataFrame(rows)


def main() -> int:
    config = load_config()
    ds = config["dataset"]
    df = pd.read_csv(REPO_ROOT / ds["cleaned_file"], dtype=str, keep_default_na=False)
    raw = pd.read_csv(REPO_ROOT / ds["annotation_file"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assignments = pd.read_csv(REPO_ROOT / config["splitting"]["assignments_file"], dtype=str)
    df = df.merge(assignments[["sample_id", "split"]], on="sample_id", how="left")
    if df["split"].isna().any():
        print("ABORT: samples without a canonical split")
        return 1
    subs = ds["secondary_label_columns"]
    load_categories(dataset_columns=list(raw.columns))   # validates config/categories.yaml

    OUT.mkdir(parents=True, exist_ok=True)
    suff = sufficiency(df, subs)
    tables = {
        "all_dataset_labels.csv": label_inventory(df, raw, subs),
        "label_mapping_analysis.csv": mapping_table(subs),
        "category_sufficiency.csv": suff,
        "subcategory_structure.csv": subcategory_structure(df, subs),
        "modality_support.csv": modality_support(df, suff),
    }
    for name, table in tables.items():
        table.to_csv(OUT / name, index=False, lineterminator="\n")
    print(suff[["category", "positives", "positives_train", "positives_validation", "feasibility"]].to_string(index=False))
    print(f"Wrote {len(tables)} tables to {OUT.relative_to(REPO_ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
