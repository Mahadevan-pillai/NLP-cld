"""Research Dashboard: the STORED results of Tasks 5-15 (read-only; nothing typed by hand, nothing recomputed)."""

import streamlit as st

from src.application import research

st.title("Research Dashboard")
st.markdown("All numbers and figures below are read from the project's stored result files (`results/`). "
            "**VALIDATION** results guided development; **TEST** results come from the single final evaluation of "
            "frozen models (Task 13) and the post-hoc OCR / explainability studies (Tasks 14-15).")


def show_table(key: str, columns: list[str] | None = None, note: str | None = None) -> None:
    df = research.table(key, columns)
    title = research.ARTIFACTS[key]["title"]
    if df is None:
        st.info(f"{title}: result file unavailable ({research.ARTIFACTS[key]['path']}).")
        return
    st.markdown(f"**{title}**")
    st.dataframe(df, hide_index=True)
    st.caption(f"Source: `{research.ARTIFACTS[key]['path']}`" + (f" · {note}" if note else ""))


def show_figure(key: str, width: int | None = None) -> None:
    p = research.figure(key)
    if p is None:
        st.info(f"{research.ARTIFACTS[key]['title']}: figure unavailable.")
        return
    st.image(str(p), caption=f"{research.ARTIFACTS[key]['title']} · {research.ARTIFACTS[key]['path']}",
             width=width or "stretch")


METRICS = ["accuracy", "precision", "recall", "macro_f1", "weighted_f1"]
tabs = st.tabs(["Dataset", "Development (validation)", "Final comparison (TEST)", "Ablations", "OCR robustness",
                "Error analysis & XAI"])
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        show_table("class_distribution")
        show_table("split_distribution", ["split", "samples", "count_non_misogynistic", "count_misogynistic",
                                          "samples_with_text"])
    with c2:
        show_figure("fig_class_distribution")
    show_table("dataset_summary")
with tabs[1]:
    show_table("text_baselines", ["experiment_id", "model", "preprocessing", *METRICS], "VALIDATION")
    show_table("text_models", ["experiment_id", "model", "preprocessing", *METRICS], "VALIDATION")
    show_table("image_models", ["experiment_id", "encoder", "classifier", *METRICS], "VALIDATION")
    show_table("multimodal_models", ["experiment_id", "model", "modality", *METRICS], "VALIDATION")
    show_table("tuning", ["family", "baseline_model", "tuned_model", "baseline_macro_f1", "tuned_macro_f1",
                          "absolute_change"], "VALIDATION (Task 12)")
    show_figure("fig_tuning", 720)
with tabs[2]:
    show_table("main_results", ["system", "model", "modality", *METRICS, "macro_f1_ci95"], "TEST, evaluated once")
    show_figure("fig_final_macro_f1", 760)
    show_table("rq_answers")
    st.markdown("**Confusion matrices (TEST)**")
    cols = st.columns(4)
    for col, key in zip(cols, ("fig_cm_T05", "fig_cm_I03", "fig_cm_M04", "fig_cm_B07")):
        with col:
            show_figure(key)
with tabs[3]:
    show_table("preprocessing_ablation", ["family", "preprocessing", "model", "macro_f1", "macro_f1_ci95"], "TEST")
    show_figure("fig_preprocessing_ablation", 720)
    show_table("modality_ablation", ["population", "modality", "model", *METRICS], "TEST")
    show_table("fusion_ablation", ["model", "description", "macro_f1", "macro_f1_ci95", "diff_vs_M04", "diff_vs_M04_ci95"], "TEST")
    show_figure("fig_modality_comparison", 720)
with tabs[4]:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        show_table("ocr_quality", note="OCR text vs the dataset caption (itself EasyOCR output: agreement, not accuracy)")
    with c2:
        show_table("ocr_robustness", ["model", "condition", "macro_f1", "drop_vs_reference", "drop_ci95",
                                      "predictions_changed_vs_reference"], "TEST, frozen models")
    c3, c4 = st.columns(2)
    with c3:
        show_figure("fig_ocr_f1")
    with c4:
        show_figure("fig_ocr_cer")
with tabs[5]:
    show_figure("fig_error_distribution", 720)
    show_table("error_distribution", ["table", "model", "category", "subcategory", "errors", "percent",
                                      "false_positives", "false_negatives"], "TEST errors; manual categories are one reviewer's judgement")
    show_table("gender_term_shortcut", note="false-positive rate on non-misogynistic TEST memes (Fisher exact, Holm)")
    show_table("modality_agreement")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("fig_modality_disagreement")
    with c2:
        show_figure("fig_text_attribution")
    st.caption("Explanations show model attribution and may not reflect human reasoning or causality.")
