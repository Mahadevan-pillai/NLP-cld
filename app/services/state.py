"""Streamlit caching around the application services (Task 16).

- get_audit(): the start-up integration audit (checksums against the Task 13 freeze), once per process;
- get_service(): AnalysisService with every available model loaded ONCE (st.cache_resource), so clicks
  never reload models;
- get_history(): HistoryService (lazy MongoDB connection);
- db_status(): cached for 60 s so the sidebar does not ping MongoDB on every rerun.
Failures are turned into safe messages; nothing here contains model or database logic.
"""

import time

import streamlit as st

from src.application import artifacts
from src.application.history import HistoryService
from src.application.service import AnalysisService
from src.application.settings import load_app_config


@st.cache_resource(show_spinner="Checking models and artifacts (first start only)...")
def get_audit() -> dict:
    t0 = time.perf_counter()
    try:
        rows = artifacts.audit()
        return {"rows": rows, "seconds": round(time.perf_counter() - t0, 2), "error": None}
    except Exception as exc:
        return {"rows": [], "seconds": round(time.perf_counter() - t0, 2),
                "error": f"The start-up check failed ({type(exc).__name__})."}


@st.cache_resource(show_spinner="Loading the final models (first start only; about 20-30 s on CPU)...")
def get_service():
    audit = get_audit()
    if audit["error"]:
        return None
    t0 = time.perf_counter()
    svc = AnalysisService(audit_rows=audit["rows"])
    try:
        svc.warm_timings = svc.warm_up()
        svc.warm_error = None
    except Exception as exc:
        svc.warm_timings, svc.warm_error = {}, f"Model loading failed ({type(exc).__name__})."
    svc.load_seconds = round(time.perf_counter() - t0, 2)
    return svc


@st.cache_resource
def get_history() -> HistoryService:
    return HistoryService()


@st.cache_data(ttl=60, show_spinner=False)
def db_status() -> dict:
    return get_history().status()


@st.cache_data(show_spinner=False)
def app_config() -> dict:
    return load_app_config()
