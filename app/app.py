"""Streamlit entry point (Task 16). Run from the repository root:

    streamlit run app/app.py

UI only: every prediction, explanation, moderation decision and database call goes through
src/application (AnalysisService / HistoryService); no model or PyMongo code lives in app/.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st  # noqa: E402

from components import ui  # noqa: E402
from services import state  # noqa: E402

st.set_page_config(page_title="Hinglish Misogynistic Meme Detection", page_icon=":material/fact_check:",
                   layout="wide", initial_sidebar_state="expanded")

pages = [
    st.Page("pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("pages/analyze_text.py", title="Analyze Text", icon=":material/notes:"),
    st.Page("pages/analyze_meme.py", title="Analyze Meme", icon=":material/image:"),
    st.Page("pages/history.py", title="Analysis History", icon=":material/history:"),
    st.Page("pages/dashboard.py", title="Research Dashboard", icon=":material/insights:"),
    st.Page("pages/about.py", title="Feedback / About", icon=":material/info:"),
]
nav = st.navigation(pages)
ui.sidebar_status(state)
nav.run()
