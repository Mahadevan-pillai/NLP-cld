"""Analyze Text: validation -> Task 6 preprocessing -> FINAL_TEXT_MODEL -> explanation -> moderation -> history."""

import streamlit as st

from components import ui
from services import state

cfg = state.app_config()
st.title("Analyze Text")
st.markdown("Enter English, Hinglish (romanised Hindi) or Hindi text. The **text model** (T05, fine-tuned MuRIL) "
            "predicts whether it is potentially misogynistic; a TF-IDF baseline is used as a second opinion for the "
            "moderation recommendation.")

svc = state.get_service()
if svc is None or getattr(svc, "warm_error", None):
    st.error("The models could not be loaded, so text analysis is unavailable. See Home → System status.")
    st.stop()
ok, blocking = svc.availability["text"]
if not ok:
    st.error(f"Text analysis is unavailable: missing or changed components: {', '.join(blocking)}. "
             "No other model is substituted.")
    st.stop()

with st.form("text_form"):
    text = st.text_area("Text to analyse", height=140, max_chars=cfg["text_input"]["max_chars"],
                        placeholder="e.g. a caption or a comment in English / Hinglish")
    consent = ui.consent_box("text_consent", cfg["history"]["consent_default"])
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    with st.spinner("Analysing the text (prediction + explanation)..."):
        result = svc.analyze_text(text, explain=cfg["xai"]["text_explanation"] == "automatic")
        saved = state.get_history().save(result, consent) if result["status"] != "error" else None
    st.session_state["text_result"], st.session_state["text_saved"] = result, saved

result = st.session_state.get("text_result")
if result:
    if result["status"] == "error":
        ui.show_error(result)
    else:
        ui.warnings(result)
        ui.final_card(result)
        with st.container(border=True):
            st.markdown("**Input**")
            st.text(result["input"]["original_text"])
            st.markdown(f"**Normalised text** (what the model reads; {result['input']['preprocessing']})")
            st.text(result["input"]["normalized_text"] or "(empty after preprocessing)")
        st.markdown("##### MODEL OUTPUTS")
        c1, c2 = st.columns(2)
        with c1:
            ui.model_card("Text model (primary)", result["models"]["text"])
        with c2:
            ui.model_card("TF-IDF baseline (second opinion)", result["models"]["second_opinion"])
        st.markdown("##### EXPLAIN PREDICTION")
        ui.text_explanation((result.get("explanation") or {}).get("text"))
        ui.disclaimer()
        ui.timings(result)
        saved = st.session_state.get("text_saved")
        ui.history_message(saved)
        if saved and saved.get("saved"):
            ui.feedback_form(state.get_history(), saved["analysis_id"], f"text_{saved['analysis_id']}")
