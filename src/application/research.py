"""Read-only access to the STORED research results for the dashboard (Task 16).

Every number shown in the dashboard is read from these files (Tasks 5-15); nothing is typed by hand and
nothing is recomputed. A missing file returns None so the page can show "unavailable" instead of crashing.
Only aggregate tables / figures are listed: files that contain dataset text are deliberately absent.
"""

from pathlib import Path

import pandas as pd

from src.utils.config import REPO_ROOT

ARTIFACTS = {
    # ---- dataset (Tasks 4-7)
    "dataset_summary": {"path": "results/tables/eda/dataset_summary.csv", "section": "Dataset", "title": "Dataset summary (MIMIC, cleaned)"},
    "class_distribution": {"path": "results/tables/eda/class_distribution.csv", "section": "Dataset", "title": "Class distribution"},
    "split_distribution": {"path": "results/tables/splits/split_distribution.csv", "section": "Dataset", "title": "Train / validation / test split"},
    "fig_class_distribution": {"path": "results/figures/eda/01_class_distribution.png", "section": "Dataset", "title": "Class distribution"},
    # ---- development results (validation; Tasks 8-12)
    "text_baselines": {"path": "results/tables/text_baselines/baseline_results.csv", "section": "Text models", "title": "TF-IDF baselines B01-B06 (VALIDATION)"},
    "text_models": {"path": "results/tables/text_models/text_model_comparison.csv", "section": "Text models", "title": "MuRIL text models T01-T05 (VALIDATION)"},
    "image_models": {"path": "results/tables/image_models/image_model_results.csv", "section": "Image model", "title": "Image models I01-I02 (VALIDATION)"},
    "multimodal_models": {"path": "results/tables/multimodal/main_comparison.csv", "section": "Multimodal model", "title": "Fusion models M01-M03 (VALIDATION)"},
    "tuning": {"path": "results/tables/tuning/before_after_tuning.csv", "section": "Hyperparameter tuning", "title": "Before / after tuning (VALIDATION)"},
    "fig_tuning": {"path": "results/figures/tuning/before_after_tuning.png", "section": "Hyperparameter tuning", "title": "Before / after tuning"},
    # ---- final TEST results (Task 13, evaluated once)
    "main_results": {"path": "results/tables/final/main_results.csv", "section": "Final comparison", "title": "Final TEST results (758 memes)"},
    "rq_answers": {"path": "results/tables/final/research_question_answers.csv", "section": "Final comparison", "title": "Research-question answers"},
    "fig_final_macro_f1": {"path": "results/figures/final/final_macro_f1.png", "section": "Final comparison", "title": "Macro-F1 of the final models (TEST, 95% CI)"},
    "fig_cm_T05": {"path": "results/figures/final/T05_test_confusion_matrix.png", "section": "Final comparison", "title": "T05 text model"},
    "fig_cm_I03": {"path": "results/figures/final/I03_test_confusion_matrix.png", "section": "Final comparison", "title": "I03 image model"},
    "fig_cm_M04": {"path": "results/figures/final/M04_test_confusion_matrix.png", "section": "Final comparison", "title": "M04 multimodal model"},
    "fig_cm_B07": {"path": "results/figures/final/B07_test_confusion_matrix.png", "section": "Final comparison", "title": "B07 TF-IDF baseline"},
    "preprocessing_ablation": {"path": "results/tables/final/preprocessing_ablation.csv", "section": "Preprocessing ablation", "title": "RAW / BASIC / HINGLISH (TEST)"},
    "fig_preprocessing_ablation": {"path": "results/figures/final/preprocessing_ablation.png", "section": "Preprocessing ablation", "title": "Preprocessing ablation"},
    "modality_ablation": {"path": "results/tables/final/modality_ablation.csv", "section": "Modality ablation", "title": "Text vs image vs multimodal (TEST)"},
    "fusion_ablation": {"path": "results/tables/final/fusion_ablation.csv", "section": "Modality ablation", "title": "Fusion ablation (TEST)"},
    "fig_modality_comparison": {"path": "results/figures/final/modality_comparison.png", "section": "Modality ablation", "title": "Modality comparison"},
    # ---- OCR robustness (Task 14)
    "ocr_quality": {"path": "results/tables/ocr/ocr_quality_summary.csv", "section": "OCR robustness", "title": "OCR quality on TEST"},
    "ocr_robustness": {"path": "results/tables/ocr/ocr_robustness_results.csv", "section": "OCR robustness", "title": "Frozen models with OCR text (TEST)"},
    "fig_ocr_f1": {"path": "results/figures/ocr/reference_vs_ocr_macro_f1.png", "section": "OCR robustness", "title": "Reference caption vs OCR text"},
    "fig_ocr_cer": {"path": "results/figures/ocr/cer_distribution.png", "section": "OCR robustness", "title": "CER distribution"},
    # ---- explainability and errors (Task 15)
    "error_distribution": {"path": "results/tables/explainability/error_distribution.csv", "section": "Error analysis", "title": "Error taxonomy (TEST errors)"},
    "gender_term_shortcut": {"path": "results/tables/explainability/gender_term_shortcut.csv", "section": "Error analysis", "title": "Female-reference-word shortcut"},
    "modality_agreement": {"path": "results/tables/explainability/modality_agreement_summary.csv", "section": "Error analysis", "title": "Modality agreement (TEST)"},
    "fig_error_distribution": {"path": "results/figures/explainability/error_category_distribution.png", "section": "Error analysis", "title": "Primary category of M04 errors"},
    "fig_modality_disagreement": {"path": "results/figures/explainability/modality_disagreement.png", "section": "Error analysis", "title": "Agreement and modality dependence"},
    "fig_text_attribution": {"path": "results/figures/explainability/text_attribution_example.png", "section": "Error analysis", "title": "Text attribution on synthetic probes"},
}
SECTIONS = ["Dataset", "Text models", "Image model", "Multimodal model", "Hyperparameter tuning", "Final comparison",
            "Preprocessing ablation", "Modality ablation", "OCR robustness", "Error analysis"]


def path(key: str) -> Path | None:
    p = REPO_ROOT / ARTIFACTS[key]["path"]
    return p if p.exists() else None


def table(key: str, columns: list[str] | None = None) -> pd.DataFrame | None:
    """The stored CSV (optionally a column subset), or None if it is missing / unreadable."""
    p = path(key)
    if p is None:
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep] if keep else df
    return df


def figure(key: str) -> Path | None:
    return path(key)


def section_items(section: str) -> list[str]:
    return [k for k, v in ARTIFACTS.items() if v["section"] == section]
