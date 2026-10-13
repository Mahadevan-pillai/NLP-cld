"""Application service layer (Task 16): everything the Streamlit app (and a future API) needs, without UI code.

- settings.py: config/app.yaml + the frozen final-model selection from config.yaml
- artifacts.py: start-up audit of every component (integration matrix); never substitutes a model
- validation.py: text and upload validation (model-compatible limits, safe messages)
- service.py: AnalysisService.analyze_text() / analyze_meme() / explain_*() - orchestrates the existing
  preprocessing, inference, OCR, explainability and moderation code; no model logic is re-implemented
- history.py: HistoryService - consent-based MongoDB history + feedback through the Task 2 repositories;
  the app keeps working when MongoDB is unavailable
- research.py: read-only loaders for the stored research results (dashboard)
- logging_setup.py: safe application logging (no user text, no credentials)
"""
