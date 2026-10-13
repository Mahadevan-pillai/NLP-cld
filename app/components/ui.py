"""Presentation helpers for the Streamlit pages (Task 16). No model, OCR or database logic here.

Wording rules (Task 15): "model prediction", "potentially misogynistic", "flagged for review",
"model attribution"; scores are labelled as UNCALIBRATED; the model classifies content, not people.
Every piece of user-provided text rendered as HTML is escaped first.
"""

import html

import streamlit as st

MOD_STYLE = {"ALLOW": ("green", ":material/check_circle:", "ALLOW"),
             "WARNING": ("orange", ":material/warning:", "WARNING"),
             "HUMAN_REVIEW": ("blue", ":material/person_search:", "HUMAN REVIEW"),
             None: ("gray", ":material/block:", "NO RECOMMENDATION")}
DISCLAIMER = ("Explanations show model attribution and may not reflect human reasoning or causality. "
              "Predictions are outputs of statistical models trained on one dataset; they can be wrong. "
              "The model classifies content, not people.")


def sidebar_status(state) -> None:
    cfg = state.app_config()
    with st.sidebar:
        st.caption(f"Research prototype · v{cfg['app_version']}")
        audit = state.get_audit()
        if audit["error"]:
            st.error(audit["error"])
        else:
            from src.application.artifacts import mode_availability
            for mode, (ok, blocking) in mode_availability(audit["rows"]).items():
                label = "Analyze Text" if mode == "text" else "Analyze Meme"
                if ok:
                    st.markdown(f":green[:material/check_circle:] {label} ready")
                else:
                    st.markdown(f":red[:material/error:] {label} unavailable")
        db = state.db_status()
        st.markdown(f":green[:material/database:] History: MongoDB connected" if db["ok"]
                    else ":orange[:material/database:] History unavailable (analysis still works)")
        st.divider()
        st.caption("Scores are uncalibrated model outputs, not confidence percentages.")


def disclaimer() -> None:
    st.caption(f":material/info: {DISCLAIMER}")


def show_error(result: dict) -> None:
    err = result.get("error") or {}
    if err.get("category") == "invalid_input":
        st.warning(err.get("message", "The input could not be analysed."), icon=":material/edit_note:")
    else:
        st.error(err.get("message", "The analysis failed."), icon=":material/error:")


def warnings(result: dict) -> None:
    for w in result.get("warnings", []):
        st.info(w, icon=":material/info:")


def _badge(display: str, label: int) -> str:
    color = "red" if label == 1 else "green"
    return f":{color}-badge[{display}]"


def final_card(result: dict) -> None:
    """FINAL ANALYSIS card: prediction, category, moderation, score and version."""
    p, mod, cat = result["prediction"], result["moderation"] or {}, result["category"]
    color, icon, text = MOD_STYLE.get(mod.get("recommendation"), MOD_STYLE[None])
    with st.container(border=True):
        st.markdown("##### FINAL ANALYSIS")
        c1, c2, c3 = st.columns([1.3, 1.3, 1])
        with c1:
            st.markdown("**Prediction**")
            st.markdown(_badge(p["display"], p["label"]))
            st.caption(f"from the {p['source']} · {p['model_version']}")
        with c2:
            st.markdown("**Category**")
            st.markdown(cat["label"] or "—")
            if cat.get("subcategories"):
                st.caption(f"Sub-categories: {cat['subcategories']}")
        with c3:
            st.markdown("**Moderation**")
            st.markdown(f":{color}-badge[{icon} {text}]")
            st.caption(mod.get("reason", ""))
        st.caption(f"Model score (uncalibrated): {p['score']:.3f} · {p['score_type']}")
        if mod.get("wording"):
            st.caption(f"{mod['wording']} No content is blocked automatically.")


def model_card(title: str, pred: dict | None) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if not pred:
            st.caption("not run")
            return
        if "unavailable" in pred:
            st.caption(pred["unavailable"])
            return
        st.markdown(_badge(pred["display"], pred["label"]))
        st.caption(f"{pred['model_version']} · input: {pred['used_input']}")
        st.caption(f"score (uncalibrated): {pred['score']:.3f}")


def model_comparison(result: dict) -> None:
    st.markdown("##### MODEL COMPARISON")
    m = result["models"]
    cols = st.columns(3)
    with cols[0]:
        model_card("Text model", m["text"])
    with cols[1]:
        model_card("Image model", m["image"])
    with cols[2]:
        model_card("Multimodal model", m["multimodal"])
    mod = result.get("modality")
    if mod:
        st.markdown(f"**Modality agreement:** `{mod['agreement']}`"
                    + (f" (disagreeing: {', '.join(mod['disagreeing'])} model)" if mod["disagreeing"] else ""))
        st.caption(f"Inside the multimodal model: {mod['dependence']} (one modality replaced by its training "
                   "average). This is a label-flip check, not a percentage contribution.")


def token_highlights(tokens: list[dict]) -> None:
    """Words coloured by attribution: red = towards 'misogynistic', blue = towards 'not misogynistic'."""
    if not tokens:
        st.caption("No words to highlight (the processed text is empty).")
        return
    peak = max(abs(t["contribution"]) for t in tokens) or 1.0
    spans = []
    for t in tokens:
        a = min(1.0, abs(t["contribution"]) / peak)
        rgb = "214,69,65" if t["contribution"] > 0 else "49,130,206"
        tip = f"{t['contribution']:+.3f} log-odds"
        spans.append(f'<span title="{tip}" style="background: rgba({rgb},{0.12 + 0.6 * a:.2f}); padding: 2px 4px; '
                     f'margin: 2px; border-radius: 4px; display: inline-block">{html.escape(t["token"])}</span>')
    st.markdown("<div style='line-height: 2.1'>" + " ".join(spans) + "</div>", unsafe_allow_html=True)
    st.caption("Red: pushed the text model towards 'potentially misogynistic'. Blue: towards 'not flagged'. "
               "Stronger colour = larger attribution (hover for the value).")


def text_explanation(expl: dict | None, title: str = "Text signals (text model attribution)") -> None:
    if not expl:
        return
    st.markdown(f"**{title}**")
    if expl.get("status") != "ok":
        st.caption(expl.get("message", "Explanation unavailable."))
        return
    token_highlights(expl["tokens"])
    top = [f"`{html.escape(t['token'])}` ({t['contribution']:+.2f})" for t in expl["top_positive"][:5]]
    if top:
        st.caption("Strongest words towards 'potentially misogynistic': " + ", ".join(top))
    st.caption(f"Method: Integrated Gradients ({expl['method'].get('steps')} steps) · completeness error "
               f"{expl['completeness_error']:+.3f} · {expl['duration_ms'] / 1000:.1f} s")


def image_explanation(expl: dict | None) -> None:
    if not expl:
        return
    st.markdown("**Image regions (image model attribution)**")
    if expl.get("status") != "ok":
        st.caption(expl.get("message", "Explanation unavailable."))
        return
    st.image(expl["overlay"], caption="Red regions pushed the image model towards 'potentially misogynistic', blue "
                                      "regions towards 'not flagged' (occlusion, coarse 4 x 4 grid).", width=420)
    st.caption("The image model also reacts to text printed inside the picture, so a highlighted region may be "
               f"the meme's text. Computed in {expl['duration_ms'] / 1000:.1f} s.")


def consent_box(key: str, default: bool) -> bool:
    return st.checkbox("Save this analysis to history (stores the text / OCR text, predictions and image size; "
                       "never the image itself)", value=default, key=key)


def history_message(saved: dict | None) -> None:
    if not saved:
        return
    if saved["saved"]:
        st.success(f"{saved['message']} (ID {saved['analysis_id'][:8]})", icon=":material/save:")
    elif not saved["ok"]:
        st.warning(f"{saved['message']} The prediction above is still valid.", icon=":material/cloud_off:")
    else:
        st.caption(saved["message"])


def feedback_form(history, analysis_id: str, key: str) -> None:
    with st.form(key=f"fb_{key}", clear_on_submit=True, border=True):
        st.markdown("**Was this prediction correct?** (feedback is stored for research; it never retrains the model)")
        rating = st.radio("Rating", ["correct", "incorrect"], horizontal=True, label_visibility="collapsed")
        comment = st.text_area("Optional comment", max_chars=500, height=70)
        if st.form_submit_button("Send feedback"):
            r = history.add_feedback(analysis_id, rating, comment)
            (st.success if r["ok"] else st.warning)(r["message"])


def timings(result: dict) -> None:
    t = result.get("timings_ms", {})
    st.caption(" · ".join(f"{k}: {v / 1000:.2f} s" for k, v in t.items()))
