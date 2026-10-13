"""Analysis History: recent saved analyses (only those saved with consent), filters, details and feedback."""

from datetime import datetime, time, timezone

import pandas as pd
import streamlit as st

from components import ui
from services import state

cfg = state.app_config()
st.title("Analysis History")
st.markdown("Analyses appear here only if the user ticked **Save this analysis to history**. Images are never stored.")

history = state.get_history()
c1, c2, c3 = st.columns(3)
kind = c1.selectbox("Type", ["all", "text", "meme"])
label = c2.selectbox("Final prediction", ["all", "misogynistic", "non_misogynistic"],
                     format_func=lambda v: {"all": "all", "misogynistic": cfg["labels"]["misogynistic"],
                                            "non_misogynistic": cfg["labels"]["non_misogynistic"]}[v])
since = c3.date_input("Since", value=None)
res = history.recent(cfg["history"]["list_limit"], None if kind == "all" else kind, None if label == "all" else label,
                     datetime.combine(since, time.min, tzinfo=timezone.utc) if since else None)
if not res["ok"]:
    st.warning(res["message"], icon=":material/cloud_off:")
    st.stop()
items = res["items"]
if not items:
    st.info("No saved analyses match these filters.")
    st.stop()


def row(d: dict) -> dict:
    final = (d.get("predictions") or {}).get("final") or {}
    return {"time (UTC)": d["created_at"].strftime("%Y-%m-%d %H:%M"), "type": d["analysis_type"],
            "prediction": cfg["labels"].get(final.get("label"), final.get("label", "")),
            "category": "misogyny (binary)", "moderation": (d.get("moderation") or {}).get("recommendation") or "",
            "models": " / ".join((d.get("model_versions") or {}).values()), "source": d.get("source", ""),
            "id": d["analysis_id"]}


table = pd.DataFrame([row(d) for d in items])
st.dataframe(table.drop(columns=["id"]), hide_index=True)
st.caption(f"{len(items)} most recent analyses (newest first). Content is shown only in the details view.")

choice = st.selectbox("Show details for", table["id"], format_func=lambda i: f"{table.set_index('id').loc[i, 'time (UTC)']} · "
                      f"{table.set_index('id').loc[i, 'type']} · {i[:8]}")
detail = history.get(choice)
if not detail["ok"]:
    st.warning(detail["message"])
    st.stop()
d = detail["item"]
with st.container(border=True):
    st.markdown(f"**Analysis {d['analysis_id'][:8]}** · {d['analysis_type']} · app {d.get('app_version', '?')}")
    preds = d.get("predictions") or {}
    st.dataframe(pd.DataFrame([{"model output": k, "label": cfg["labels"].get(v["label"], v["label"]),
                                "score (uncalibrated)": round(v.get("score"), 4) if v.get("score") is not None else None}
                               for k, v in preds.items()]), hide_index=True)
    mod = d.get("moderation") or {}
    st.markdown(f"**Moderation:** {mod.get('recommendation') or '—'} · {mod.get('reason') or ''}")
    if d.get("ocr"):
        st.caption(f"OCR: {d['ocr'].get('status')} · route {d['ocr'].get('route')} · confidence {d['ocr'].get('confidence')}")
    if d.get("processing_time_ms") is not None:
        st.caption(f"Processing time: {d['processing_time_ms'] / 1000:.2f} s · models: {d.get('model_versions')}")
    with st.expander("Show stored text (may contain offensive content)"):
        inp = d.get("input") or {}
        for key in ("original_text", "ocr_text", "normalized_text"):
            if inp.get(key):
                st.markdown(f"*{key.replace('_', ' ')}*")
                st.text(inp[key])
        if inp.get("image_metadata"):
            st.caption(f"Image metadata only (no image stored): {inp['image_metadata']}")
    if detail["feedback"]:
        st.caption("Feedback: " + ", ".join(f"{f['rating']}" for f in detail["feedback"]))
ui.feedback_form(history, d["analysis_id"], f"hist_{d['analysis_id']}")
