"""Run the EDA on the cleaned primary dataset and write tables + figures.

Run from the repository root:
    python scripts/run_eda.py

Reads data/processed/<dataset>_cleaned.csv (Task 4) and the referenced
raw images. Nothing is modified. Writes:
    results/tables/eda/*.csv      statistics (aggregates and sample IDs only, no text)
    results/figures/eda/*.png     figures (300 dpi)
    results/metrics/eda_summary.json   key numbers used in docs/EDA_REPORT.md

Deterministic: no random operations.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import eda, eda_plots as plots  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402
from src.utils.io import write_json_if_changed  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables" / "eda"
FIGURES = REPO_ROOT / "results" / "figures" / "eda"
NEAR_DUP_MAX_DISTANCE = 4   # of 64 dHash bits


def save(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False, encoding="utf-8", lineterminator="\n")


def main() -> int:
    config = load_config()
    key = config["dataset"]["primary"]
    fp_path = REPO_ROOT / "results" / "metrics" / f"{key}_cleaning_fingerprint.json"
    if not (REPO_ROOT / config["dataset"]["cleaned_file"]).exists() or not fp_path.exists():
        print("EDA failed: cleaned dataset or Task 4 fingerprint missing; run scripts/clean_dataset.py first.")
        return 1

    names = eda.class_names(config)
    df = eda.load_cleaned(config)
    checks = eda.verify_cleaned(df, config, json.loads(fp_path.read_text(encoding="utf-8")))
    print(f"Input checks: {checks}")
    if not all(checks.values()):
        print("EDA aborted: input does not match the Task 4 output.")
        return 1

    secondary_cols = config["dataset"]["secondary_label_columns"]
    features = eda.text_features(df)

    # ---- statistics ----
    class_table, ratio = eda.class_distribution(df, names)
    secondary = eda.secondary_label_stats(df, secondary_cols)
    lengths = eda.length_statistics(features, names)
    vocab = eda.vocabulary_summary(df, names)
    unigrams = eda.top_terms(df, names, n=1, top=50)
    bigrams = eda.top_terms(df, names, n=2, top=30)
    distinctive = eda.distinctive_terms(df, names)
    scripts = eda.script_by_class(features, names)
    social = eda.feature_by_class(features, names)
    meta = eda.image_metadata(df)
    image_stats = eda.image_statistics(meta, names)
    resolutions = eda.common_resolutions(meta, names)
    near_dups = eda.near_duplicate_images(df, max_distance=NEAR_DUP_MAX_DISTANCE)
    handles = eda.mention_handles(df, names)
    id_pattern = eda.id_pattern_association(df)
    outliers = eda.outliers(features, meta)
    avail = eda.availability(df)
    present = features[features["has_text"]]
    dev_share = present["devanagari_letter_share"].dropna()

    class_text = present.groupby("target_label").agg(
        samples=("sample_id", "size"), mean_chars=("char_count", "mean"), median_chars=("char_count", "median"),
        mean_words=("word_count", "mean"), median_words=("word_count", "median"),
        mean_devanagari_letter_share=("devanagari_letter_share", "mean")).round(3).reset_index()
    class_text.insert(1, "label_name", class_text["target_label"].map(names))
    empty_by_class = features.loc[~features["has_text"], "target_label"].value_counts().sort_index()
    class_text["empty_text_samples"] = class_text["target_label"].map(empty_by_class).fillna(0).astype(int)

    # ---- tables ----
    summary_rows = [
        ("dataset", config["dataset"]["candidates"][0]["name"]), ("cleaned_file", config["dataset"]["cleaned_file"]),
        ("samples", len(df)), *[(f"samples_{n}", int((df.target_label == k).sum())) for k, n in names.items()],
        ("imbalance_ratio_majority_to_minority", ratio["imbalance_ratio_majority_to_minority"]),
        *[(k, v) for k, v in avail.items() if k != "samples"],
        ("vocabulary_size_eda_tokens", vocab["all"]["vocabulary_size"]), ("tokens_eda", vocab["all"]["tokens"]),
        ("images_jpeg", int((meta["format"] == "JPEG").sum())),
        (f"near_duplicate_image_pairs_dhash_le_{NEAR_DUP_MAX_DISTANCE}", len(near_dups)),
        ("near_duplicate_pairs_cross_class", int(near_dups["cross_class"].sum())),
        ("filename_pattern_perfectly_predicts_label", id_pattern["pattern_perfectly_predicts_label"]),
    ]
    save(__import__("pandas").DataFrame(summary_rows, columns=["metric", "value"]), "dataset_summary.csv")
    save(class_table, "class_distribution.csv")
    save(secondary["per_label"], "secondary_label_statistics.csv")
    save(secondary["combinations"], "secondary_label_combinations.csv")
    save(secondary["cooccurrence_within_positive"].reset_index(names="label"), "secondary_label_cooccurrence.csv")
    save(lengths, "text_statistics.csv")
    save(class_text, "class_text_statistics.csv")
    save(__import__("pandas").concat([unigrams, bigrams]), "token_frequency.csv")
    save(distinctive, "distinctive_terms.csv")
    save(scripts, "language_script_statistics.csv")
    save(social, "social_media_features.csv")
    save(image_stats, "image_statistics.csv")
    save(resolutions, "image_resolutions_by_class.csv")
    save(near_dups, "near_duplicate_images.csv")
    save(handles, "mention_handles.csv")
    save(outliers.groupby("category").agg(samples=("sample_id", "size"), rule=("rule", "first"))
         .reset_index(), "outlier_summary.csv")
    save(outliers, "outlier_samples.csv")

    # ---- figures ----
    FIGURES.mkdir(parents=True, exist_ok=True)
    f = FIGURES
    figs = [
        plots.class_distribution(class_table, f / "01_class_distribution.png"),
        plots.histogram(present["char_count"], f / "02_text_char_length_hist.png",
                        "Text length (characters)", "Characters per text", bins=50,
                        note=f"Non-empty texts (n = {len(present):,}); {len(df) - len(present)} samples have no text."),
        plots.length_by_class(features, names, f / "03_text_length_by_class.png"),
        plots.histogram(present["word_count"], f / "04_word_count_hist.png", "Word count",
                        "Whitespace-separated words per text", bins=range(0, int(present["word_count"].max()) + 2),
                        note=f"Non-empty texts (n = {len(present):,})."),
        plots.top_terms(unigrams[unigrams["group"] == "all"], f / "05_top_terms_overall.png",
                        "Most frequent terms (all samples)"),
        plots.distinctive_terms(distinctive, names, f / "06_distinctive_terms_by_class.png"),
        plots.histogram(meta["width"], f / "07_image_width_hist.png", "Image width", "Width (px)", bins=50,
                        note=f"All {len(meta):,} images."),
        plots.histogram(meta["height"], f / "08_image_height_hist.png", "Image height", "Height (px)", bins=50,
                        note=f"All {len(meta):,} images."),
        plots.histogram(meta["aspect_ratio"], f / "09_aspect_ratio_hist.png", "Image aspect ratio",
                        "Width / height (1.0 = square)", bins=60, note=f"All {len(meta):,} images."),
        plots.grouped_percent_by_class(scripts, "script_category", names, f / "10_script_categories_by_class.png",
                                       "Script / language indicators by class (HEURISTIC)", "Share of class (%)",
                                       note="Character-range and marker-word heuristic, not language identification."),
        plots.grouped_percent_by_class(social[(social["count_" + names[0]] + social["count_" + names[1]]) > 0],
                                       "feature", names, f / "11_social_media_features_by_class.png",
                                       "Text features by class", "Share of samples with text (%)",
                                       note="URLs and emojis: 0 samples (not shown). Descriptive only; no tests applied."),
        plots.secondary_combinations(secondary["combinations"], f / "12_secondary_label_combinations.png",
                                     int((df.target_label == 1).sum())),
        plots.handles_by_class(handles, names, f / "13_mention_handles_by_class.png"),
    ]

    # ---- summary JSON (numbers quoted in docs/EDA_REPORT.md) ----
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_checks": checks, "class_distribution": class_table.to_dict("records"), "imbalance": ratio,
        "availability": avail, "vocabulary": vocab,
        "secondary": {"per_label": secondary["per_label"].to_dict("records"),
                      "labels_per_positive_sample": {str(k): int(v) for k, v in secondary["labels_per_positive_sample"].items()}},
        "text_lengths": lengths.to_dict("records"), "class_text": class_text.to_dict("records"),
        "devanagari_letter_share_nonempty": eda.describe(dev_share),
        "script_categories": scripts.to_dict("records"), "social_features": social.to_dict("records"),
        "image_statistics": image_stats.to_dict("records"),
        "image_resolutions_top": resolutions.to_dict("records"),
        "image_formats": meta["format"].value_counts().to_dict(), "image_modes": meta["mode"].value_counts().to_dict(),
        "near_duplicates": {"max_distance": NEAR_DUP_MAX_DISTANCE, "pairs": len(near_dups),
                            "cross_class": int(near_dups["cross_class"].sum()),
                            "same_text": int(near_dups["same_text"].sum()),
                            "already_grouped_in_task4": int(near_dups["already_grouped_in_task4"].sum()),
                            "distance_0": int((near_dups["hamming_distance"] == 0).sum())},
        "handles": handles.to_dict("records"), "id_pattern": id_pattern,
        "outliers": outliers["category"].value_counts().sort_index().to_dict(),
        "figures": [p.name for p in figs],
    }
    status = write_json_if_changed(REPO_ROOT / "results" / "metrics" / "eda_summary.json", summary)
    print(f"Wrote {len(list(TABLES.glob('*.csv')))} tables, {len(figs)} figures; eda_summary.json {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
