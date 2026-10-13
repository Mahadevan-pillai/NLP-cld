"""Analyze Meme: upload -> validation -> OCR -> text / image / multimodal models -> moderation -> (explain) -> history.

The uploaded bytes stay in memory (this browser session only); they are never written to disk or stored in MongoDB.
"""

import hashlib

import streamlit as st

from components import ui
from services import state

cfg = state.app_config()
st.title("Analyze Meme")
st.markdown("Upload a meme (JPG or PNG). Its text is read with OCR, then three models run: the **text model** (OCR text "
            "only), the **image model** (the picture only) and the **multimodal model** (both). The multimodal "
            "prediction is the main result; if no reliable text is found, the image model is used alone.")

svc = state.get_service()
if svc is None or getattr(svc, "warm_error", None):
    st.error("The models could not be loaded, so meme analysis is unavailable. See Home → System status.")
    st.stop()
ok, blocking = svc.availability["meme"]
if not ok:
    st.error(f"Meme analysis is unavailable: missing or changed components: {', '.join(blocking)}. "
             "No other model is substituted.")
    st.stop()

up = st.file_uploader("Meme image", type=cfg["upload"]["allowed_extensions"],
                      help=f"JPG or PNG, at most {cfg['upload']['max_upload_mb']} MB. The file name is not used.")
if up is not None:
    data = up.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.get("meme_digest") != digest:              # a new upload resets the previous result
        st.session_state.update(meme_digest=digest, meme_bytes=data, meme_result=None, meme_saved=None, meme_expl=None)
    c_img, c_form = st.columns([1, 1.3])
    with c_img:
        st.image(data, caption="Uploaded meme (kept in memory only)", width="stretch")
    with c_form:
        consent = ui.consent_box("meme_consent", cfg["history"]["consent_default"])
        if st.button("Analyze", type="primary"):
            with st.spinner("Reading the text (OCR) and running the three models..."):
                result = svc.analyze_meme(data)
                saved = state.get_history().save(result, consent) if result["status"] != "error" else None
            st.session_state.update(meme_result=result, meme_saved=saved, meme_expl=None)

result = st.session_state.get("meme_result") if up is not None else None
if result:
    if result["status"] == "error":
        ui.show_error(result)
        st.stop()
    ui.warnings(result)
    ui.final_card(result)
    o = result["input"]["ocr"]
    with st.container(border=True):
        st.markdown("**OCR (text read from the image)**")
        conf = "n/a" if o["confidence"] is None else f"{o['confidence']:.2f}"
        st.caption(f"Status: {o['status']} · route: {o['route']} ({o['reason']}) · regions: {o['regions']} · "
                   f"mean OCR confidence: {conf} · {o['engine']} {o['engine_version'] or ''}")
        st.text(o["text"] or "(no text detected)")
        if o["normalized_text"]:
            st.markdown("**Normalised OCR text** (what the text model reads)")
            st.text(o["normalized_text"])
    ui.model_comparison(result)
    st.caption("The image model never sees the OCR text; the text model never sees the picture.")
    st.markdown("##### EXPLAIN PREDICTION")
    if st.button("Explain prediction", help="Image occlusion map (about 3-5 s) + word attributions of the OCR text"):
        with st.spinner("Computing explanations..."):
            st.session_state["meme_expl"] = svc.explain_meme(st.session_state["meme_bytes"], o["text"])
    expl = st.session_state.get("meme_expl")
    if expl:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            ui.image_explanation(expl["image"])
        with c2:
            ui.text_explanation(expl["text"], "Text signals in the OCR text (text model attribution)")
    ui.disclaimer()
    ui.timings(result)
    saved = st.session_state.get("meme_saved")
    ui.history_message(saved)
    if saved and saved.get("saved"):
        ui.feedback_form(state.get_history(), saved["analysis_id"], f"meme_{saved['analysis_id']}")
