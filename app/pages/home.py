"""Home page: what the system analyses, how, which categories are supported, responsible use."""

import pandas as pd
import streamlit as st

from components import ui
from services import state
from src.application.settings import supported_categories

cfg = state.app_config()
st.title(cfg["title"])
st.markdown(f"*{cfg['subtitle']}*")

st.markdown(
    "Memes combine an image with short, often **code-mixed Hindi-English (Hinglish)** text. Misogyny in memes can be "
    "carried by the words, by the picture, or only by the two together. This application runs the project's "
    "**frozen final models** on a text or a meme, explains which inputs moved each model's score, compares the "
    "text-only, image-only and multimodal predictions, and suggests a cautious moderation action.")

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("#### :material/notes: Text analysis")
    st.markdown("English / Hinglish / Hindi text → Task 6 preprocessing → fine-tuned MuRIL text model → word "
                "attributions → moderation recommendation.")
    st.page_link("pages/analyze_text.py", label="Analyze Text", icon=":material/arrow_forward:")
with c2, st.container(border=True):
    st.markdown("#### :material/image: Meme analysis")
    st.markdown("Uploaded meme → OCR (EasyOCR) → text model + image model (SigLIP) + multimodal fusion → "
                "comparison, explanation and moderation. The **multimodal** prediction is the main result.")
    st.page_link("pages/analyze_meme.py", label="Analyze Meme", icon=":material/arrow_forward:")

st.markdown("#### How it works")
st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=10];
  meme [label="Meme image"]; text [label="Text"]; ocr [label="OCR (EasyOCR)"]; pre [label="Task 6 preprocessing\\n(Hinglish normalisation)"];
  tm [label="Text model\\nT05 fine-tuned MuRIL"]; im [label="Image model\\nI03 SigLIP + LR"]; mm [label="Multimodal model\\nM04 fusion MLP"];
  xai [label="Explanations\\n(IG / occlusion / ablation)"]; mod [label="Moderation\\nALLOW / WARNING / HUMAN REVIEW"];
  meme -> ocr -> pre; text -> pre; pre -> tm; meme -> im; pre -> mm; meme -> mm;
  tm -> xai; im -> xai; mm -> xai; xai -> mod;
}""")

st.markdown("#### Supported classification")
cats = supported_categories()
for task in cats["enabled"]:
    st.markdown(f"- **{', '.join(task['categories'])}** — {task['type']} task (misogynistic / not misogynistic). "
                f"{cats['label_semantics']}")
for task in cats["disabled"]:
    st.markdown(f"- Not available: {', '.join(task['categories'])} ({task['type']} sub-categories; annotated in the "
                "dataset but no model has been trained).")
with st.expander("Categories this system does NOT detect"):
    st.dataframe(pd.DataFrame([{"category": k, "why not": v} for k, v in cats["unsupported"].items()]),
                 hide_index=True)

st.markdown("#### Responsible use")
st.warning("This is an academic research prototype. Predictions can be wrong and can reflect shortcuts learned from "
           "the training data (for example, the text model over-reacts to words referring to women). Use the output "
           "as a model prediction to support human review, never as proof that content or a person is misogynistic. "
           "Nothing is blocked automatically.", icon=":material/policy:")
ui.disclaimer()

with st.expander("System status (start-up integration check)"):
    audit = state.get_audit()
    if audit["error"]:
        st.error(audit["error"])
    else:
        st.dataframe(pd.DataFrame(audit["rows"]).drop(columns=["preprocessing"], errors="ignore"),
                     hide_index=True)
        st.caption(f"Checked in {audit['seconds']} s (artifacts compared with the Task 13 freeze manifest).")
