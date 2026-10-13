"""Feedback / About: objective, architecture, data, models, research questions, method, limitations, responsible use."""

import pandas as pd
import streamlit as st

from components import ui
from services import state
from src.application import research
from src.application.settings import final_models

cfg = state.app_config()
st.title("Feedback / About")

st.markdown("#### Feedback")
st.markdown("After an analysis is **saved to history** (with your consent), you can mark the prediction as *correct* or "
            "*incorrect* and add a short comment, directly below the result or on the **Analysis History** page. Feedback "
            "is stored for research review only; it never retrains or changes the models automatically.")

st.markdown("#### Objective")
st.markdown("Detect potentially misogynistic **English / Hinglish memes and texts**, compare text-only, image-only and "
            "multimodal models, study the effect of Hinglish normalisation and OCR noise, and explain the predictions "
            "where technically valid. This is a B.Tech final-year research project; results are not peer-reviewed.")

st.markdown("#### Architecture")
st.markdown("- **UI:** Streamlit (`app/`), presentation only.\n"
            "- **Services:** `src/application` - validation, the unified `AnalysisService`, consent-based history.\n"
            "- **Models and pipelines:** frozen Task 6 preprocessing, Task 14 OCR, Task 15 explainability and "
            "moderation rules, Task 2 MongoDB repositories.")

m = final_models()
st.markdown("#### Models in this application (frozen after the final evaluation)")
st.dataframe(pd.DataFrame([
    {"role": "text model (FINAL_TEXT_MODEL)", "id": m["text"], "family": "fine-tuned MuRIL (google/muril-base-cased)"},
    {"role": "image model (FINAL_IMAGE_MODEL)", "id": m["image"], "family": "frozen SigLIP-B/16 + logistic regression"},
    {"role": "multimodal model (FINAL_MULTIMODAL_MODEL)", "id": m["multimodal"], "family": "fusion MLP (T05 text + SigLIP image)"},
    {"role": "second opinion (text moderation only)", "id": m["second_opinion_text"], "family": "TF-IDF + logistic regression"},
]), hide_index=True)

st.markdown("#### Dataset")
summary = research.table("class_distribution")
st.markdown("**MIMIC** (Hindi-English code-mixed memes; binary *Misogyny* label plus Objectification / Prejudice / "
            "Humiliation sub-labels). Leakage-safe split: train / validation / test, with the test set used once.")
if summary is not None:
    st.dataframe(summary, hide_index=True)

st.markdown("#### Research questions (answers from the final TEST evaluation)")
rq = research.table("rq_answers")
st.dataframe(rq, hide_index=True) if rq is not None else st.info("Result file unavailable.")

st.markdown("#### Methodology overview")
st.markdown("Dataset inspection and cleaning → EDA → Task 6 text preprocessing (RAW / BASIC / HINGLISH) → leakage-safe "
            "split → TF-IDF baselines → MuRIL text models → SigLIP image model → fusion → validation-only tuning → "
            "model freeze → single TEST evaluation → OCR robustness → explainability and error analysis → this application.")

st.markdown("#### Limitations")
st.markdown("- One dataset (MIMIC); captions are OCR output; scores are **uncalibrated**.\n"
            "- Known shortcuts: the text model over-reacts to words referring to women and largely ignores negation; "
            "the image model reacts to pictured women; empty captions are flagged by the text model.\n"
            "- Only binary misogyny is modelled: 'not flagged' does not mean the content is safe in other respects; "
            "sub-categories are not available.\n"
            "- OCR takes about 5 s per meme on CPU and can miss stylised or small text.\n"
            "- Explanations are model attribution, not proof of reasoning.")

st.markdown("#### Responsible use")
st.warning("Use the outputs as model predictions that support human review. The system classifies content, not people; "
           "it never blocks content automatically. Do not use it to make decisions about individuals.",
           icon=":material/policy:")
ui.disclaimer()
st.caption(f"Application version {cfg['app_version']}. Research documents: docs/FINAL_EVALUATION_REPORT.md, "
           "docs/OCR_ROBUSTNESS_REPORT.md, docs/EXPLAINABILITY_ERROR_ANALYSIS_REPORT.md, docs/APPLICATION_INTEGRATION_REPORT.md.")
